socket = require "socket"
-- module common
common = {}

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local function send_wrapper(sock, data)
  local len     = #data
  local len_str = binarizer.code.uint32(len)
  if len > 256 then
    sock:send(len_str)
    sock:send(data)
  else
    sock:send(len_str..data)
  end
end

local function recv_wrapper(sock)
  local msg  = sock:receive("5")
  if not msg then return nil end
  local len  = binarizer.decode.uint32(msg)
  local data = sock:receive(len)
  return data
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local logger_methods,
logger_class_metatable = class("common.logger")

function logger_class_metatable:__call()
  local obj = {}
  return class_instance(obj,self,true)
end

function logger_methods:debug(format,...)
  fprintf(io.stderr, format, ...)
end

function logger_methods:warningf(format,...)
  fprintf(io.stderr, format, ...)
end

function logger_methods:print(...)
  print("#", ...)
end

function logger_methods:raw_print(...)
  print(...)
end

function logger_methods:printf(format,...)
  printf("#\t" .. format, ...)
end

function logger_methods:raw_printf(format,...)
  printf(format, ...)
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function common.make_connection_handler(select_handler,message_reply,
					connections)
  return function(conn,data,error_msg,partial)
    local action
    if data then
      print("# RECEIVED DATA: ", data)
      action,data = data:match("^([^%s]*)%s(.*)$")
      if message_reply[action] == nil then
	select_handler:send(conn, "UNKNOWN ACTION")
      elseif type(message_reply[action]) == "string" then
	select_handler:send(conn, message_reply[action])
      else
	local ans = message_reply[action](conn, data)
	select_handler:send(conn, ans)
      end
      if action == "EXIT" then
	-- following instruction allows chaining of several actions
	select_handler:close(conn, function() connections:mark_as_dead(conn) end)
      else
	-- following instruction allows chaining of several actions
	select_handler:receive(conn,
			       common.make_connection_handler(select_handler,
							      message_reply,
							      connections))
      end
    end
  end
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local select_methods,select_class_metatable=class("common.select_handler")

function select_class_metatable:__call(select_timeout)
  local obj =  {
    select_timeout = select_timeout,
    data = {},
    recv_query  = {},
    send_query  = {},
  }
  return class_instance(obj,self,true)
end

function select_methods:clean(conn)
  self.data[conn]       = nil
  self.recv_query[conn] = nil
  self.send_query[conn] = nil
end

function select_methods:accept(conn, func)
  local func = func or function() end
  self.recv_query[conn] = true
  self.data[conn] = self.data[conn] or {}
  table.insert(self.data[conn],
	       { op = "accept", func=func })
end

function select_methods:receive(conn, func)
  local func = func or function() end
  self.recv_query[conn] = true
  self.data[conn] = self.data[conn] or {}
  table.insert(self.data[conn],
	       { op = "receive", func=func })
end

function select_methods:send(conn, func)
  local func = func or function() end
  if type(func) == "string" then
    local str = func
    func = function() return str end
  end
  self.send_query[conn] = true
  self.data[conn] = self.data[conn] or {}
  table.insert(self.data[conn],
	       { op = "send", func=function() return func() end })
end

function select_methods:close(conn, func)
  local func = func or function() end
  if type(func) == "string" then func = function() return func end end
  self.data[conn] = self.data[conn] or {}
  table.insert(self.data[conn],
	       { op = "close", func=func })
end

local process = {
  
  receive = function(conn,func,recv_map,send_map)
    if recv_map[conn] then
      func(conn, recv_wrapper(conn) )
      recv_map[conn] = nil
      return true
    end
    return false
  end,
  
  accept = function(conn,func,recv_map,send_map)
    if recv_map[conn] then
      local new_conn = conn:accept()
      new_conn:settimeout(TIMEOUT)
      func(conn, new_conn)
      recv_map[conn] = nil
      return true
    end
    return false
  end,
  
  send = function(conn,func,recv_map,send_map)
    if send_map[conn] then
      send_wrapper(conn, func(conn) .. "\n")
      send_map[conn] = nil
      return true
    end
    return false
  end,
  
  close = function(conn,func,recv_map,send_map)
    printf("# Closing connection %s:%d\n", conn:getsockname())
    conn:close()
    func(conn)
    return "close"
  end,
}


function select_methods:execute(timeout)
  local recv_list = iterator(pairs(self.recv_query)):select(1):table()
  local send_list = iterator(pairs(self.send_query)):select(1):table()
  --
  local recv_list,send_list = socket.select(recv_list, send_list,
					    self.select_timeout)
  --
  local recv_map = iterator(ipairs(recv_list)):select(2):
  reduce(function(acc,a) acc[a] = true return acc end, {})
  --
  local send_map = iterator(ipairs(send_list)):select(2):
  reduce(function(acc,a) acc[a] = true return acc end, {})
  --
  for conn,data in pairs(self.data) do
    if #data == 0 then
      self.send_query[conn] = nil
      self.recv_query[conn] = nil
      self.data[conn] = nil
    else
      local remove = {}
      for i=1,#data do
	local d = data[i]
	local processed = process[d.op](conn,d.func,recv_map,send_map)
	if processed == "close" then 
	  remove = {}
	  self.data[conn] = nil
	  self.recv_query[conn] = nil
	  self.send_query[conn] = nil
	  break
	elseif processed then table.insert(remove, i)
	else break
	end
      end
      for j=#remove,1,-1 do table.remove(self.data[conn], remove[j]) end
      --
      iterator(ipairs(self.data[conn] or {})):select(2):field("op"):
      apply(function(v)
	      if v=="accept" or v=="receive" then
		self.recv_query[conn] = true
	      elseif v == "send" then
		self.send_query[conn] = true
	      end
	    end)
    end
  end
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local connections_set_methods,
connections_set_class_metatable = class("common.connections_set")

function connections_set_class_metatable:__call()
  local obj = { connections = {}, dead_connections = {} }
  return class_instance(obj,self,true)
end

-- creates a new coroutine and adds it to the previous table
function connections_set_methods:add(conn)
  table.insert(self.connections, conn)
end

function connections_set_methods:mark_as_dead(conn)
  self.dead_connections[conn] = true
end

function connections_set_methods:remove_dead_conections()
  -- list of connections to be removed (by index)
  local remove = iterator(ipairs(self.connections)):
  filter(function(k,v) return self.dead_connections[v] end):select(1):
  reduce(table.insert, {})
  -- remove the listed connections
  for i=#remove,1,-1 do
    local pos = remove[i]
    self.dead_connections[pos] = nil
    table.remove(self.connections, pos)
  end
end

function connections_set_methods:close()
  iterator(ipairs(self.connections)):apply(function(i,c)c:close()end)
  self.connections = {}
  self.dead_connections = {}
end
