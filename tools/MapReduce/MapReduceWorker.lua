package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
require "worker"
--
local MASTER_ADDRESS = arg[1] or error("Needs a master address")
local MASTER_PORT    = arg[2] or error("Needs a master port")
local WORKER_NAME    = arg[3] or error("Needs a worker name")
local WORKER_BIND    = arg[4] or error("Needs a worker bind address")
local WORKER_PORT    = arg[5] or error("Needs a worker bind port")
local NUMP           = tonumber(arg[6] or error("Needs a number of cores"))
local MEM            = arg[7] or error("Needs a memory number, use any suffix B,K,M,G,T,P,E,Z,Y for 1024 powers")
local suffix_powers = {
  B = 1/(1024*1024),
  K = 1/1024,
  M = 1,
  G = 1024,
  T = 1024*1024,
  P = 1024*1024*1024,
  E = 1024*1024*1024*1024,
  Z = 1024*1024*1024*1024*1024,
  Y = 1024*1024*1024*1024*1024*1024,
}
local MEM,suffix = MEM:match("([0-9]+)(.)")
if not suffix or not suffix_powers[suffix] then
  error("The memory size needs a valid suffix: B,K,M,G,T,P,E,Z,Y")
end
MEM = tonumber(MEM) * suffix_powers[suffix]
--
local BIND_TIMEOUT      = 10
local TIMEOUT           = 10  -- in seconds
local MASTER_PING_TIMER = 10  -- in seconds
local PENDING_TH        = 50  -- number of computed results
--
local MASTER_IS_ALIVE  = false
local connections      = common.connections_set()
local select_handler   = common.select_handler()
local workersock       = socket.tcp()
local logger           = common.logger()
local cores            = { }
local free_cores       = { }
local aux_free_cores_dict = { }
local mapkey2core      = { }
local core_tmp_names   = iterator(range(1,NUMP)):map(os.tmpname):table()
local map_pending_jobs = {}
local reduce_pending_jobs = {}
local pending_results  = { }
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function check_error(p,msg)
  if not p then logger:warningf("ERROR: %s\n",msg) end
  return p
end

local pending = false
function register_to_master(name,port,master_address,master_port,nump,mem)
  if pending then return end
  local s,msg = common.connections_set.connect(master_address, master_port)
  if not check_error(s,msg) then return false end
  pending = true
  select_handler:send(s, string.format("WORKER %s %s %d %f",
				       name, port, nump, mem))
  select_handler:receive(s,
			 function(conn,msg)
			   local msg = table.unpack(string.tokenize(msg or ""))
			   if msg ~= "OK" then
			     select_handler:close(s, function() pending=false end)
			   end
			 end)
  select_handler:send(s, "EXIT")
  select_handler:receive(s,
			 function(conn,msg)
			   local msg = table.unpack(string.tokenize(msg or ""))
			   if msg == "EXIT" then MASTER_IS_ALIVE = true end
			 end)
  select_handler:close(s, function() pending=false end)
end

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------
local message_reply = {
  
  PING = function(conn,name)
    local name = table.concat(string.tokenize(name or ""))
    return (name==WORKER_NAME and "PONG") or "ERROR"
  end,
  
  EXIT = "EXIT",
  
  LOADAVG = function(conn,name)
    local name = table.concat(string.tokenize(name or ""))
    if name ~= WORKER_NAME then return "ERROR" end
    local f = io.popen("uptime")
    local loadavg = f:read("*l")
    f:close()
    loadavg = loadavg:match("^.*: .* (%d.%d%d).* .*$"):gsub(",",".")
    return loadavg
  end,
  
  TASK = function(conn,msg)
    local taskid,nump,arg,script = msg:match("^%s*([^%s]+)%s*([^%s]+)%s*(return %b{})%s*(.*)$")
    cores       = {}
    free_cores  = {}
    mapkey2core = {}
    aux_free_cores_dict = {}
    map_pending_jobs    = {}
    reduce_pending_jobs = {}
    pending_results     = {}
    collectgarbage("collect")
    for i=1,tonumber(nump) do
      cores[i] = worker.core(logger,
			     core_tmp_names[i],
			     script,
			     arg,
			     taskid,
			     workersock,
			     connections,
			     WORKER_PORT)
      free_cores[i] = cores[i]
      aux_free_cores_dict[cores[i]] = true
    end
    logger:printf("Received TASK %s to be executed in %d cores\n",taskid,nump)
    return "OK"
  end,

  MAP = function(conn,msg)
    table.insert(map_pending_jobs, msg)
    return "EXIT"
  end,

  REDUCE = function(conn,msg)
    table.insert(reduce_pending_jobs, msg)
    return "EXIT"
  end,

  SHARE = function(conn,msg)
    for i=1,#cores do cores[i]:share(msg) end
    return "EXIT"
  end,
}

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function check_master(select_handler)
  if not MASTER_IS_ALIVE then
    logger:warningf("Master is dead?\n")
    register_to_master(WORKER_NAME, WORKER_PORT,
		       MASTER_ADDRESS, MASTER_PORT,
		       NUMP, MEM)    
    return false
  else
    logger:debug("Master is alive, ping")
    local s,error_msg = common.connections_set.connect(MASTER_ADDRESS,
						       MASTER_PORT)
    if not check_error(s,error_msg) then
      MASTER_IS_ALIVE=false
      return false
    end
    --
    select_handler:send(s, "PING " .. WORKER_NAME)
    select_handler:receive(s,
			   function(conn, msg)
			     local msg = table.unpack(string.tokenize(msg or ""))
			     if msg ~= "PONG" then
			       MASTER_IS_ALIVE = false
			       select_handler:close(s)
			     end
			   end)
    select_handler:send(s, "EXIT")
    select_handler:receive(s)
    select_handler:close(s)
  end
  return true
end

function worker_func(workersock,conn)
  if conn then
    local a,b = conn:getsockname()
    local c,d = conn:getpeername()
    logger:debugf("Connection received at %s:%d from %s:%d\n",a,b,c,d)
    connections:add(conn)
    select_handler:receive(conn, common.make_connection_handler(select_handler,
								message_reply,
								connections))
  end
  -- following instruction allows action chains
  select_handler:accept(workersock, worker_func)
end

function append_result(c, t)
  local result = c:read_result()
  if result then table.insert(t, result) end
end

function execute_pending_job(pending_jobs, method, cache)
  if #pending_jobs > 0 then
    local idx   = 1
    local cache = cache or {}
    while #free_cores > 0 and #pending_jobs > 0 do
      local msg = pending_jobs[idx]
      local str = msg:match("^%s*(return .*)$")
      key,value = common.load(str,logger)
      local c = cache[key]
      if not c then
	c = table.remove(free_cores, 1)
	cache[key] = c
	aux_free_cores_dict[c] = nil
      end
      if not c:busy() then
	c[method](c, str)
	c:flush()
	table.remove(pending_jobs, idx)
      else idx = idx + 1
      end
    end
  end
end

function process_pending_results(pending_results)
  -- CONNECTION WITH MASTER
  local conn = common.connections_set.connect(MASTER_ADDRESS, MASTER_PORT)
  -- TODO: check error
  --if not check_error(ok,error_msg) then return false end
  for _,result in ipairs(pending_results) do
    select_handler:send(conn, result)
    -- TODO: throw error if ok~="EXIT"
    select_handler:receive(conn)
  end
  select_handler:send(conn, "EXIT")
  select_handler:receive(conn)
  select_handler:close(conn)
  return {}
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  logger:printf("Running worker with %d cores and %fM memory, %s registred "..
		  "to master %s:%s and binded to %s:%s\n",
		NUMP, MEM,
		WORKER_NAME, MASTER_ADDRESS, MASTER_PORT,
		WORKER_BIND, WORKER_PORT)
  
  local ok,msg = workersock:bind(WORKER_BIND, WORKER_PORT)
  while not ok do
    logger:warningf("ERROR: %s\n", msg)
    util.sleep(BIND_TIMEOUT)
    ok,msg = workersock:bind(WORKER_BIND, WORKER_PORT)
  end
  ok,msg = workersock:listen()
  if not ok then error(msg) end
  
  -- register SIGINT handler for safe worker stop
  signal.register(signal.SIGINT,
		  function()
		    logger:raw_print("\n# Closing worker")
		    connections:close()
		    if workersock then workersock:close() workersock = nil end
		    iterator(ipairs(core_tmp_names)):select(2):apply(os.remove)
		    collectgarbage("collect")
		    os.exit(0)
		  end)

  logger:print("Ok")

  -- appends accept
  select_handler:accept(workersock, worker_func)  
  
  local clock = util.stopwatch()
  clock:go()
  check_master(select_handler)
  while true do
    collectgarbage("collect")
    local cpu,wall = clock:read()
    if wall > MASTER_PING_TIMER then
      check_master(select_handler)
      clock:stop()
      clock:reset()
      clock:go()
    end
    if #cores > 0 then
      for i=1,#cores do
	local c = cores[i]
	if not aux_free_cores_dict[c] and not c:busy() then
	  aux_free_cores_dict[c] = true
	  table.insert(free_cores, c)
	  append_result(c, pending_results)
	end
      end
      local n = #pending_results
      if n > 0 then
	if n > PENDING_TH or #cores == #free_cores then
	  pending_results = process_pending_results(pending_results)
	end
      end
      execute_pending_job(map_pending_jobs, "do_map", mapkey2core)
      execute_pending_job(reduce_pending_jobs, "do_reduce")
    end
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    connections:remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
