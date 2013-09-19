package.path = string.format("%s?.lua;%s", string.get_path(arg[0]), package.path)
--
require "MapReduceCommon"
worker = MapReduceCommon.worker
--
local MASTER_BIND = arg[1] or "*"
local MASTER_PORT = arg[2] or "8888"
--
local TIMEOUT           = 1    -- in seconds
local WORKER_PING_TIMER = 15   -- in seconds
--

-- function map(key, value) do stuff coroutine.yield(key, value) end
-- function reduce(key, iterator) do stuff return result end

local workers          = {} -- a table with registered workers
local inv_workers      = {} -- the inverted dictionary
local connections      = {} -- a table with running connections
local dead_connections = {} -- a table with dead connections
-- handler for I/O
local select_handler   = MapReduceCommon.select_handler(WORKER_PING_TIMER)
local master           = socket.tcp() -- the main socket

---------------------------------------------------------------
------------------- CONNECTION HANDLER ------------------------
---------------------------------------------------------------

local message_reply = {
  PING = "PONG",
  EXIT = "EXIT",
  WORKER =
    function(address,name,port)
      print("# Received WORKER action: ", address, name, port)
      if inv_workers[name] then
	print("# Updating WORKER data: ", address, name, port)
	inv_workers[name]:update(address,port)
      else
	table.insert(workers, worker(name,address,port))
	inv_workers[name] = workers[#workers]
      end
      return "OK"
    end,
}

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

function check_workers(t, inv_t)
  local dead_workers = iterator(ipairs(t)):
  -- filter the dead ones
  filter(function(i,w) return w:dead() end):
  table(function(IDX,i,w) return IDX,i end)
  --
  iterator(ipairs(dead_workers)):
  -- removes dead workers
  apply(function(_,i)
	  local data = table.pack(t[i]:get())
	  print("# Removing dead WORKER: ", table.unpack(data))
	  table.remove(t,i)
	  inv_t[data[1]] = nil
	end)
end

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
      local ans = message_reply[action](conn:getsockname(), table.unpack(data))
      select_handler:send(conn, ans)
    end
    if action == "EXIT" then
      -- following instruction allows chaining of several actions
      select_handler:close(conn, function() mark_as_dead(conn) end)
    else
      -- following instruction allows chaining of several actions
      select_handler:receive(conn, connection_handler)
    end
  end
end

function remove_dead_conections()
  -- list of connections to be removed (by index)
  local remove = iterator(ipairs(connections)):
  filter(function(k,v) return dead_connections[v] end):select(1):
  reduce(table.insert, {})
  -- remove the listed connections
  for i=#remove,1,-1 do
    local pos = remove[i]
    dead_connections[pos] = nil
    table.remove(connections, pos)
  end
end

function master_func(master,conn)
  if conn then
    local a,b = conn:getsockname()
    local c,d = conn:getpeername()
    printf ("# Connection received at %s:%d from %s:%d\n",a,b,c,d)
    add_connection(conn)
    select_handler:receive(conn, connection_handler)
  end
  -- following instruction allows action chains
  select_handler:accept(master, master_func)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function main()
  printf("# Running master binded to %s:%s\n", MASTER_BIND, MASTER_PORT)
  
  master:bind(MASTER_BIND, MASTER_PORT)
  master:listen()
  master:settimeout(TIMEOUT)
  
  -- register SIGINT handler for safe master stop
  signal.register(signal.SIGINT,
		  function()
		    print("\n# Closing master")
		    iterator(ipairs(connections)):apply(function(i,c)c:close()end)
		    if master then master:close() master = nil end
		    collectgarbage("collect")
		    os.exit(0)
		  end)
  
  print("# Ok")
  
  -- appends accept
  select_handler:accept(master, master_func)

  local clock = util.stopwatch()
  clock:go()
  while true do
    collectgarbage("collect")
    local cpu,wall = clock:read()
    if wall > WORKER_PING_TIMER then
      for i=1,#workers do workers[i]:ping(select_handler, TIMEOUT) end
      clock:stop()
      check_workers(workers, inv_workers)
      clock:reset()
      clock:go()
    end
    -- execute pending operations
    select_handler:execute(TIMEOUT)
    remove_dead_conections()
  end
end

------------------------------------------------------------------------------

main()
