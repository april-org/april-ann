package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
--
require "master"
require "common"
--
local MASTER_BIND = arg[1] or "*"
local MASTER_PORT = arg[2] or "8888"
--
local BIND_TIMEOUT      = 10
local TIMEOUT           = 10   -- in seconds
local WORKER_PING_TIMER = 10   -- in seconds
--

-- function map(key, value) do stuff coroutine.yield(key, value) end
-- function reduce(key, iterator) do stuff return result end

local workers          = {} -- a table with registered workers
local inv_workers      = {} -- the inverted dictionary
-- handler for I/O
local select_handler   = common.select_handler(WORKER_PING_TIMER)
local connections      = common.connections_set()
local mastersock       = socket.tcp() -- the main socket
local logger           = common.logger()

---------------------------------------------------------------
---------------------------------------------------------------
---------------------------------------------------------------

local task       = nil
local task_count = 0
function make_task_id(name)
  task_count = task_count+1
  return string.format("%s-%09d",name,task_count)
end

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------

local message_reply = {
  OK   = function() return end,
  
  PING = "PONG",
  
  EXIT = "EXIT",
  
  -- A task is defined in a Lua script. This script must be a path to a filename
  -- located in a shared disk between cluster nodes.
  -- TODO: allow to send a Lua code string, instead of a filename path
  TASK = function(conn,msg)
    local name,arg,script = msg:match("^%s*([^%s]+)%s*(return %b{})%s*(.*)")
    local address = conn:getsockname()
    logger:print("Received TASK action:", address, arg)
    if task ~= nil then
      logger:warningf("The cluster is busy\n")
      return "ERROR"
    end
    local ID = make_task_id(name)
    task = master.task(logger, select_handler, conn, ID, script, arg)
    if not task then return "ERROR" end
    return ID
  end,

  WORKER =
    function(conn,msg)
      local name,port,nump,mem = table.unpack(string.tokenize(msg or ""))
      local address = conn:getsockname()
      logger:debug("Received WORKER action:", address, name, port, nump, mem)
      local w = inv_workers[name]
      if w then
	logger:print("Updating WORKER")
	w:update(address,port,nump,mem)
      else
	logger:print("Creating WORKER")
	local w = master.worker(name,address,port,nump,mem)
	table.insert(workers, w)
	inv_workers[name] = workers[#workers]
	w:ping(select_handler, TIMEOUT)
      end
      return "OK"
    end,

  MAP_RESULT = function(conn,msg)
    local map_key,map_result = msg:match("^%s*(%b{})%s*(return .*)$")
    local address = conn:getsockname()
    map_key = common.load(string.sub(map_key,2,#map_key-1),logger)
    logger:debug("Received MAP_RESULT action: ", address, map_key)
    -- TODO: throw error if result is not well formed??
    local ok = task:process_map_result(map_key,map_result)
    -- TODO: throw error
    -- if not ok then return "ERROR" end
    return "OK"
  end,

  REDUCE_RESULT = function(conn,msg)
    local key_and_value = msg:match("^%s*(return .*)$")
    local key,value = common.load(key_and_value,logger)
    -- TODO: throw error if result is not well formed??
    local ok = task:process_reduce_result(key,value)
    -- TODO: throw error
    -- if not ok then return "ERROR" end
    return "OK"
  end,

}

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function check_workers(t, inv_t)
  local dead_workers = iterator(ipairs(t)):
  -- filter the dead ones
  filter(function(i,w) return w:dead() end):
  -- take the index
  table(function(IDX,i,w) return IDX,i end)
  --
  -- removes dead workers
  for i=#dead_workers,1,-1 do
    local data = table.pack(t[i]:get())
    logger:print("Removing dead WORKER: ", table.unpack(data))
    table.remove(t,i)
    inv_t[data[1]] = nil
  end
end

function master_func(mastersock,conn)
  if conn then
    local a,b = conn:getsockname()
    local c,d = conn:getpeername()
    logger:debugf("Connection received at %s:%d from %s:%d\n",a,b,c,d)
    connections:add(conn)
    select_handler:receive(conn,
			   common.make_connection_handler(select_handler,
							  message_reply,
							  connections))
  end
  -- following instruction allows action chains
  select_handler:accept(mastersock, master_func)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  logger:printf("Running master binded to %s:%s\n", MASTER_BIND, MASTER_PORT)
  
  local ok,msg = mastersock:bind(MASTER_BIND, MASTER_PORT)
  while not ok do
    logger:warningf("ERROR: %s\n", msg)
    util.sleep(BIND_TIMEOUT)
    ok,msg = mastersock:bind(MASTER_BIND, MASTER_PORT)
  end
  ok,msg = mastersock:listen()
  if not ok then error(msg) end
  mastersock:settimeout(TIMEOUT)
  
  -- register SIGINT handler for safe master stop
  signal.register(signal.SIGINT,
		  function()
		    logger:raw_print("\n# Closing master")
		    connections:close()
		    if master then mastersock:close() mastersock = nil end
		    collectgarbage("collect")
		    os.exit(0)
		  end)
  
  logger:print("Ok")
  
  -- appends accept
  select_handler:accept(mastersock, master_func)

  local clock = util.stopwatch()
  clock:go()
  while true do
    collectgarbage("collect")
    local cpu,wall = clock:read()
    if wall > WORKER_PING_TIMER then
      --
      iterator(ipairs(workers)):
      select(2):
      filter(function(w)return not w:dead() end):
      apply(function(w) w:ping(select_handler, TIMEOUT) end)
      --
      check_workers(workers, inv_workers)
      clock:stop()
      clock:reset()
      clock:go()
    end
    if task then
      local state = task:get_state()
      if state == "ERROR" then task = nil
      elseif state == "STOPPED" then
	if #workers > 0 then
	  -- TODO: throw an error
	  task:prepare_map_plan(workers)
	end
      elseif state == "PREPARED" then
	task:do_map(workers)
      elseif state == "MAP_FINISHED" then
	task:do_reduce(workers)
      elseif state == "REDUCE_FINISHED" then
	task:do_sequential(workers)
      elseif state == "SEQUENTIAL_FINISHED" then
	task:do_loop()
      elseif state == "FINISHED" then
	task = nil
      elseif state ~= "MAP" and state ~= "REDUCE" and state ~= "SEQUENTIAL" and state ~= "LOOP" then
	logger:warningf("Unknown task state: %s\n", state)
      end
      select_handler:execute(0.1)
    else
      select_handler:execute(TIMEOUT)
    end
    connections:remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
