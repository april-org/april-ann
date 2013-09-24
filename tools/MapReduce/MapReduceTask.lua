package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "common"
--
local conf = common.load_configuration("/etc/APRIL-ANN-MAPREDUCE/task.lua")
--
if #arg < 1 then
  error("Incorrect syntax, execute the program with at least 1 arguments:  "..
	  "TASK_SCRIPT  [ SCRIPT ARGUMENTS ]")
end
--
local BIND_TIMEOUT      = conf.bind_timeout or 10
local TIMEOUT           = conf.timeout      or 1   -- in seconds
local MASTER_PING_TIMER = conf.ping_timer   or 10   -- in seconds
--
local MASTER_ADDRESS = assert(conf.master_address, "Needs a master_address field at conf file")
local MASTER_PORT    = assert(conf.master_port, "Needs a master_port field at conf file")
local TASK_SCRIPT    = table.remove(arg, 1)
--
local logger         = common.logger(false,io.stderr)
--
task              = assert(common.load(TASK_SCRIPT,logger,table.unpack(arg)),
			   "Error loading the script")
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
NAME              = assert(task.name, "Needs a name field at the task file")
--
local aux = io.open(TASK_SCRIPT)
local TASK_SCRIPT_CONTENT = aux:read("*a")
aux:close()
--
local MASTER_CONN = nil -- persistent connection with the master
local TASK_ID = nil
local MASTER_IS_ALIVE  = false
local connections      = common.connections_set()
local select_handler   = common.select_handler()

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------

local LOOP_RETURN = true
local FINISHED    = false

local message_reply = {
  ERROR = function(conn,msg)
    FINISHED = true
    logger:warningf("ERROR, master says: %s\n",msg)
    return "OK"
  end,

  PING = "PONG",
  
  EXIT = function() FINISHED=true return "EXIT" end,

  -- TODO: Execute sequential function in a forked process?
  SEQUENTIAL = function(conn,msg)
    local map_reduce_result = common.load(msg,logger)
    if not map_reduce_result then return "ERROR" end
    -- SEQUENTIAL
    local shared = sequential(map_reduce_result) -- it is supposed to be fast
    shared = common.tostring(shared)
    return string.format("SEQUENTIAL_DONE return %s", shared)
  end,
  
  -- TODO: Execute loop function in a forked process?
  LOOP = function(conn,msg)
    -- TODO: check the TASKID
    local ret = loop() -- it is supposed to be a fast
    LOOP_RETURN = ret
    if ret then return "return true"
    else return "return false"
    end
  end,
}

------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function check_error(p,msg)
  if not p then logger:warningf("ERROR: %s\n",msg) end
  return p
end

local attempts = 0
local pending = false
function send_task_to_master(name,master_address,master_port,arg,script)
  if pending then return end
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  if not check_error(s,error_msg) then return false end
  ok,error_msg=s:connect(master_address,master_port)
  if not check_error(ok,error_msg) then return false end
  pending = true
  select_handler:send(s,
		      function()
			logger:print("Sending TASK")
			return string.format("TASK %s return %s %s",
					     name, common.tostring(arg), script)
		      end)
  select_handler:receive(s,
			 function(conn,msg)
			   local msg = table.unpack(string.tokenize(msg or ""))
			   if msg == "ERROR" then
			     select_handler:close(s, function() pending=false end)
			   else
			     attempts = 0
			     TASK_ID = msg
			     MASTER_IS_ALIVE = true
			     select_handler:send(s, "OK")
			     select_handler:receive(conn,
						    common.make_connection_handler(select_handler,
										   message_reply,
										   connections))

			   end
			 end)
  return s
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function check_master(select_handler)
  if not MASTER_IS_ALIVE then
    connections = common.connections_set()
    MASTER_CONN = send_task_to_master(NAME, MASTER_ADDRESS, MASTER_PORT, arg, TASK_SCRIPT_CONTENT)
    logger:warningf("Master is dead?\n")
    attemts = attempts + 1
    if MASTER_CONN then connections:add(MASTER_CONN) end
    return false
  else
    local ok,error_msg,s,data = true
    s,error_msg = socket.tcp()
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
			     else
			       logger:warningf("Master is alive, ping\n")
			       MASTER_IS_ALIVE = true
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
  logger:debugf("Running task %s, at master %s:%d\n",
		NAME, MASTER_ADDRESS, MASTER_PORT)
  
  -- TODO: control SIGINT signal
  -- register SIGINT handler for safe worker stop
  signal.register(signal.SIGINT,
   		  function()
		    logger:raw_print("\n# Closing worker")
		    connections:close()
		    collectgarbage("collect")
  		    os.exit(0)
  		  end)

  logger:print("Ok")

  local clock = util.stopwatch()
  clock:go()
  check_master(select_handler)
  while LOOP_RETURN and not FINISHED do
    local cpu,wall = clock:read()
    if wall > MASTER_PING_TIMER then
      collectgarbage("collect")
      check_master(select_handler)
      clock:stop()
      clock:reset()
      clock:go()
    end
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    connections:remove_dead_conections()
    if attempts > 20 then break end
  end
  while not FINISHED do
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    connections:remove_dead_conections()
  end
  select_handler:execute(TIMEOUT)
end

------------------------------------------------------------------------------

main()
