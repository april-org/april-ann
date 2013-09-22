package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
if #arg < 3 then
  error("Incorrect syntax, execute the program with at least 4 arguments: "..
	"NAME  MASTER_ADDRESS  MASTER_PORT  TASK_SCRIPT")
end
--
local BIND_TIMEOUT      = 10
local TIMEOUT           = 1    -- in seconds
local MASTER_PING_TIMER = 10   -- in seconds
--
local NAME           = table.remove(arg,1)
local MASTER_ADDRESS = table.remove(arg,1)
local MASTER_PORT    = table.remove(arg,1)
local TASK_SCRIPT    = table.remove(arg,1)
--
local logger         = common.logger()
--
task              = common.load(TASK_SCRIPT,logger,table.unpack(arg)) or error("Error loading the script")
data              = task.data   or error("Needs a data table")
mmap              = task.map    or error("Needs a map function")
mreduce           = task.reduce or error("Needs a reduce function")
decode            = task.decode or function(...) return ... end
sequential        = task.sequential or function(...) print(...) end
shared            = task.shared or function(...) return ... end
loop              = task.loop or function() return false end
split             = task.split or
  function(data,data_size,first,last)
    return data,data_size
  end
--
local aux = io.open(TASK_SCRIPT)
local TASK_SCRIPT_CONTENT = aux:read("*a")
aux:close()
--
local MASTER_CONN = nil -- persistent connection with the master
local TASK_ID = nil
local MASTER_IS_ALIVE  = false
local connections      = common.connections_set()
local select_handler   = common.select_handler(MASTER_PING_TIMER)
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function check_error(p,msg)
  if not p then logger:warningf("ERROR: %s\n",msg) end
  return p
end

local pending = false
function send_task_to_master(name,master_address,master_port,arg,script)
  if pending then return end
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  if not check_error(s,error_msg) then return false end
  ok,error_msg=s:connect(master_address,master_port)
  if not check_error(ok,error_msg) then return false end
  pending = true
  select_handler:send(s, string.format("TASK %s return %s %s",
				       name, table.tostring(arg), script))
  select_handler:receive(s,
			 function(conn,msg)
			   local msg = table.unpack(string.tokenize(msg or ""))
			   if msg == "ERROR" then
			     select_handler:close(s, function() pending=false end)
			   else
			     TASK_ID = msg
			     MASTER_IS_ALIVE = true
			   end
			 end)
  return s
end

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------

local message_reply = {
  PING = "PONG",
  
  EXIT = "EXIT",

  -- TODO: Execute sequential function in a forked process?
  SEQUENTIAL = function(conn,msg)
    local map_reduce_result = common.load(msg,logger)
    if not map_reduce_result then return "ERROR" end
    -- SEQUENTIAL
    local shared = sequential(map_reduce_result) -- it is supposed to be fast
    shared = (type(shared)=="table" and table.tostring(shared)) or tostring(shared)
    return string.format("SEQUENTIAL_DONE return %s", shared)
  end,
  
  -- TODO: Execute loop function in a forked process?
  LOOP = function(conn,msg)
    -- TODO: check the TASKID
    local ret = loop() -- it is supposed to be a fast
    if ret then return "return true"
    else return "return false"
    end
  end,
}

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function check_master(select_handler, timeout)
  if not MASTER_IS_ALIVE then
    connections = common.connections_set()
    logger:warningf("Master is dead\n")
    MASTER_CONN = send_task_to_master(NAME, MASTER_ADDRESS, MASTER_PORT, arg, TASK_SCRIPT_CONTENT)
    connections:add(MASTER_CONN, common.make_connection_handler(select_handler,
								message_reply,
								connections))
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
    select_handler:send(s, "PING " .. NAME)
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

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  logger:printf("Running task %s, at master %s:%d\n",
		NAME, MASTER_ADDRESS, MASTER_PORT)
  
  -- TODO: control SIGINT signal
  -- -- register SIGINT handler for safe worker stop
  -- signal.register(signal.SIGINT,
  -- 		  function()
  -- 		    logger:raw_print("\n# Closing worker")
  -- 		    connections:close()
  -- 		    if workersock then workersock:close() workersock = nil end
  -- 		    collectgarbage("collect")
  -- 		    os.exit(0)
  -- 		  end)

  logger:print("Ok")

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
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    connections:remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
