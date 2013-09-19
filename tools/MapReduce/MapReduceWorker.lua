package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
require "MapReduceCommon"
--
local MASTER_ADDRESS = arg[1] or error("Needs a master address")
local MASTER_PORT    = arg[2] or error("Needs a master port")
local WORKER_NAME    = arg[3] or error("Needs a worker name")
local WORKER_BIND    = arg[4] or error("Needs a worker bind address")
local WORKER_PORT    = arg[5] or error("Needs a worker bind port")
--
local TIMEOUT           =  1  -- in seconds
local MASTER_PING_TIMER = 15  -- in seconds
--

local MASTER_IS_ALIVE  = false
local connections      = {} -- a table with running connections
local dead_connections = {} -- a table with dead connections
local select_handler   = MapReduceCommon.select_handler(MASTER_PING_TIMER)
local worker           = socket.tcp()
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

local function check_error(p,msg)
  if not p then error("ERROR: "..msg) end
  return p
end

function register_to_master(address,port)
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  if not check_error(s,error_msg) then return false end
  ok,error_msg=s:connect(address,port)
  if not check_error(ok,error_msg) then return false end
  select_handler:send(s, string.format("WORKER %s %s",
				       WORKER_NAME, WORKER_PORT))
  select_handler:receive(s,
			 function(conn,msg)
			   if msg ~= "OK" then
			     select_handler:close(s, function() end)
			   end
			 end)
  select_handler:send(s, "EXIT")
  select_handler:receive(s,
			 function(conn,msg)
			   if msg == "EXIT" then MASTER_IS_ALIVE = true end
			 end)
  select_handler:close(s)
end

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------
local message_reply = {
  PING = "PONG",
  EXIT = "EXIT",
}

function connection_handler(conn,data,error_msg,partial)
  local action
  if data then
    print("# RECEIVED DATA: ", data)
    data = string.tokenize(data)
    action = table.remove(data, 1)
    if message_reply[action] == nil then
      select_handler:send(conn, "UNKNOWN ACTION")
    elseif type(message_reply[action]) == "string" then
      select_handler:send(conn, message_reply[action])
    else
      local ans = message_reply[action](conn, table.unpack(data))
      select_handler:send(conn, ans)
    end
    if action == "EXIT" then
      -- following instruction allows chaining of several actions
      select_handler:close(conn, function(conn) mark_as_dead(conn) end)
    else
      -- following instruction allows chaining of several actions
      select_handler:receive(conn, connection_handler)
    end
  end
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-- creates a new coroutine and adds it to the previous table
function add_connection(conn)
  table.insert(connections, conn)
end
function mark_as_dead(conn)
  dead_connections[conn] = true
end

function check_master()
  if not MASTER_IS_ALIVE then
    fprintf(io.stderr,"Master is dead\n")
    return false
  end
  return true
end

function remove_dead_conections()
  -- list of connections to be removed (by index)
  local remove = iterator(ipairs(connections)):
  filter(function(k,v) return dead_connections[v] end):select(1):
  reduce(table.insert, {})
  -- remove the listed connections
  table.sort(remove)
  for i=#remove,1,-1 do
    local pos = remove[i]
    dead_connections[pos] = nil
    table.remove(connections, pos)
  end
end

function worker_func(master,conn)
  if conn then
    local a,b = conn:getsockname()
    local c,d = conn:getpeername()
    printf ("# Connection received at %s:%d from %s:%d\n",a,b,c,d)
    add_connection(conn)
    select_handler:receive(conn, connection_handler)
  end
  -- following instruction allows action chains
  select_handler:accept(worker, worker_func)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  printf("# Running worker %s registred to master %s:%s and binded to %s:%s\n",
	 WORKER_NAME, MASTER_ADDRESS, MASTER_PORT,
	 WORKER_BIND, WORKER_PORT)
  register_to_master(MASTER_ADDRESS, MASTER_PORT)
  
  worker:bind(WORKER_BIND, WORKER_PORT)
  worker:listen()
  worker:settimeout(TIMEOUT)
  
  -- register SIGINT handler for safe worker stop
  signal.register(signal.SIGINT,
		  function()
		    print("\n# Closing worker")
		    iterator(ipairs(connections)):apply(function(i,c)c:close()end)
		    if master then master:close() master = nil end
		    collectgarbage("collect")
		    os.exit(0)
		  end)

  print("# Ok")

  -- appends accept
  select_handler:accept(worker, worker_func)  
  
  local clock = util.stopwatch()
  clock:go()
  while true do
    collectgarbage("collect")
    local cpu,wall = clock:read()
    if wall > MASTER_PING_TIMER then check_master() end
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
