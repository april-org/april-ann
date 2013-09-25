package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
require "worker"
--
local conf = common.load_configuration("/etc/APRIL-ANN-MAPREDUCE/worker.lua")
--
local MASTER_ADDRESS = conf.master_address or "localhost"
local MASTER_PORT    = conf.master_port or 8888
local WORKER_NAME    = conf.name or io.popen("hostname"):read("*l")
local WORKER_BIND    = conf.bind_address or '*'
local WORKER_PORT    = conf.port or 4000
local NUMP           = conf.nump or 1
local MEM            = conf.mem  or "4G"
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
local BIND_TIMEOUT      = conf.bind_timeout or 10
local TIMEOUT           = conf.timeout      or 60  -- in seconds
local MASTER_PING_TIMER = conf.ping_timer   or 60  -- in seconds
local PENDING_TH        = conf.th or 50  -- number of computed results
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
local reduce_pending_jobs  = {}
local pending_results      = { }
local reduce_ready         = false
iterator(ipairs(core_tmp_names)):apply(function(i,v)os.execute("rm -f "..v)end)
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
  if not s then return false end
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
message_reply = {
  
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
    loadavg = loadavg:match("^.*: (%d.%d%d).* .* .*$"):gsub(",",".")
    return loadavg
  end,
  
  TASK = function(conn,msg)
    local taskid,nump,arg,script = msg:match("^%s*([^%s]+)%s*([^%s]+)%s*(return %b{})%s*(.*)$")
    cores       = {}
    free_cores  = {}
    mapkey2core = {}
    aux_free_cores_dict  = {}
    map_pending_jobs     = {}
    reduce_pending_jobs  = {}
    pending_results      = {}
    reduce_ready         = false
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
    return "OK"
  end,

  REDUCE = function(conn,msg)
    table.insert(reduce_pending_jobs, msg)
    return "OK"
  end,

  REDUCE_READY = function(conn,msg)
    reduce_ready = true
    return "OK"
  end,

  SHARE = function(conn,msg)
    for i=1,#cores do cores[i]:share(msg) end
    return "OK"
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
			     else
			       MASTER_IS_ALIVE = true
			       logger:warningf("Master is alive, ping\n")
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

function send_result_to_master(c)
  -- TODO: check error
  local result = c:read_result()
  if result then
    local conn = common.connections_set.connect(MASTER_ADDRESS, MASTER_PORT)
    select_handler:send(conn, string.format("BUNCH %s", result))
    select_handler:receive(conn)
    select_handler:send(conn, "EXIT")
    select_handler:receive(conn)
    -- TODO: throw error if not received "EXIT"
    select_handler:close(conn)
  end
end

function execute_pending_map(pending_jobs, cache)
  if #pending_jobs > 0 then
    local idx   = 1
    local cache = cache or {}
    while #free_cores > 0 and #pending_jobs >= idx do
      local core,str = pending_jobs[idx]:match("%s*(%d+)%s*(return .*)")
      core = tonumber(core)
      key,value = common.load(str,logger)
      local c = cores[core]
      cache[key] = c
      if not c:busy() and not c:read_pending() then
	-- print("EXECUTING MAP", key, core, c:busy(), aux_free_cores_dict[c])
	c:do_map(str)
	c:flush()
	table.remove(pending_jobs, idx)
	if aux_free_cores_dict[c] then
	  aux_free_cores_dict[c] = nil
	end
      else idx = idx + 1
      end
    end
    free_cores = iterator(pairs(aux_free_cores_dict)):select(1):table()
  end
end

function execute_pending_reduce(pending_jobs)
  if #pending_jobs > 0 and #free_cores > 0 then
    local core_i = 0
    local N = #free_cores
    for i=1,N do
      local c = free_cores[i]
      c:begin_reduce_bunch()
      aux_free_cores_dict[c] = nil
    end
    if N > 0 then
      for idx=1,#pending_jobs do
	core_i = core_i + 1
	local str = pending_jobs[idx]
	local c   = free_cores[core_i]
	c:append_reduce_bunch(str)
	core_i = core_i % N
      end
      table.clear(pending_jobs)
    end
    for i=1,N do
      local c = free_cores[i]
      c:end_reduce_bunch()
      c:flush()
    end
    free_cores = {}
  end
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
    local cpu,wall = clock:read()
    if wall > MASTER_PING_TIMER then
      collectgarbage("collect")
      check_master(select_handler)
      clock:stop()
      clock:reset()
      clock:go()
    end
    if #cores > 0 then
      for i=1,#cores do
	local c = cores[i]
	--print(c, aux_free_cores_dict[c], c:busy())
	if not c:busy() and c:read_pending() then
	  --print("SENDING!!!!")
	  aux_free_cores_dict[c] = true
	  table.insert(free_cores, c)
	  send_result_to_master(c)
	end
      end
      execute_pending_map(map_pending_jobs, mapkey2core)
      --print(reduce_ready, #reduce_pending_jobs, #free_cores)
      if reduce_ready or #reduce_pending_jobs > PENDING_TH * #free_cores then
	execute_pending_reduce(reduce_pending_jobs)
      end
    end
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    connections:remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
