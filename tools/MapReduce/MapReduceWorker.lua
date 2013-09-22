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
local MEM            = tonumber(arg[7] or error("Needs a memory number in GB"))
--
local BIND_TIMEOUT      = 10
local TIMEOUT           =  1  -- in seconds
local MASTER_PING_TIMER = 10  -- in seconds
--
local MASTER_IS_ALIVE  = false
local connections      = common.connections_set()
local select_handler   = common.select_handler(MASTER_PING_TIMER)
local workersock       = socket.tcp()
local logger           = common.logger()
local cores            = { }
local free_cores       = { }
local aux_free_cores_dict = { }
local mapkey2core      = { }
local core_tmp_names   = iterator(range(1,NUMP)):map(os.tmpname):table()
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
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  if not check_error(s,error_msg) then return false end
  ok,error_msg=s:connect(master_address,master_port)
  if not check_error(ok,error_msg) then return false end
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
    local taskid,script = msg:match("^([^%s]+) (.*)$")
    cores = {}
    free_cores = {}
    mapkey2core = {}
    aux_free_cores_dict = {}
    collectgarbage("collect")
    for i=1,NUMP do
      cores[i] = worker.core(logger,
			     core_tmp_names[i],
			     script,
			     taskid)
      free_cores[i] = cores[i]
      aux_free_cores_dict[cores[i]] = true
    end
  end,

  MAP = function(conn,msg)
    local mapkey,job = msg:match("^([^%s]+) (.*)$")
    -- job = common.load(job)
    -- if not job then
    --  return "ERROR"
    -- end
    local c = mapkey2core[mapkey]
    if not c then
      -- TODO: error if not free cores
      c = table.remove(free_cores, 1)
      mapkey2core[mapkey] = c
      aux_free_cores_dict[c] = nil
    end
    c:write(string.format("MAP %s %s",
			  mapkey, job))
    c:flush()
    return nil
  end,

  REDUCE = function(conn,msg)
    local key,values = msg:match("^([^%s]+) (.*)$")
    -- job = common.load(job)
    -- if not job then
    --  return "ERROR"
    -- end
    -- TODO: error if not free cores
    local c = table.remove(free_cores, 1)
    aux_free_cores_dict[c] = nil
    c:write(string.format("MAP %s %s",
			  mapkey, job))
    c:flush()
    return nil
  end,
}

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function check_master(select_handler, timeout)
  if not MASTER_IS_ALIVE then
    logger:warningf("Master is dead\n")
    register_to_master(WORKER_NAME, WORKER_PORT,
		       MASTER_ADDRESS, MASTER_PORT,
		       NUMP, MEM)    
    return false
  else
    logger:print("Master is alive, ping")
    local ok,error_msg,s,data = true
    s,error_msg = socket.tcp()
    s:settimeout(timeout)
    if not check_error(s,error_msg) then MASTER_IS_ALIVE=false return false end
    ok,error_msg=s:connect(MASTER_ADDRESS,MASTER_PORT)
    if not check_error(ok,error_msg) then
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

function worker_func(master,conn)
  if conn then
    local a,b = conn:getsockname()
    local c,d = conn:getpeername()
    logger:printf("Connection received at %s:%d from %s:%d\n",a,b,c,d)
    connections:add(conn)
    select_handler:receive(conn, common.make_connection_handler(select_handler,
								message_reply,
								connections))
  end
  -- following instruction allows action chains
  select_handler:accept(workersock, worker_func)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  logger:printf("Running worker %s registred to master %s:%s and binded to %s:%s\n",
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
  workersock:settimeout(TIMEOUT)
  
  -- register SIGINT handler for safe worker stop
  signal.register(signal.SIGINT,
		  function()
		    logger:raw_print("\n# Closing worker")
		    connections:close()
		    if workersock then workersock:close() workersock = nil end
		    collectgarbage("collect")
		    os.exit(0)
		  end)

  logger:print("Ok")

  -- appends accept
  select_handler:accept(workersock, worker_func)  
  
  local clock = util.stopwatch()
  clock:go()
  check_master(select_handler, TIMEOUT)
  while true do
    collectgarbage("collect")
    local cpu,wall = clock:read()
    if wall > MASTER_PING_TIMER then
      check_master(select_handler, TIMEOUT)
      clock:stop()
      clock:reset()
      clock:go()
    end
    if #cores > 0 then
      for i=1,#cores do
	local c = cores[i]
	if not aux_free_cores_dict[c] and not cores:busy() then
	  aux_free_cores_dict[c] = true
	  table.insert(free_cores, c)
	end
      end
      -- execute pending operations
      select_handler:execute(0.1)
    else
      -- execute pending operations
      select_handler:execute(TIMEOUT)
    end
    connections:remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
