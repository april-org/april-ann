socket = require "socket"
-- module MapReduceCommon

MapReduceCommon = {}

local select_methods,select_class_metatable=class("MapReduceCommon.select_handler")

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
	       { op = "send", func=func })
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
      func(conn, conn:receive("*l"))
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
      conn:send( func(conn) .. "\n")
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


------------------------------------------------------------
------------------- WORKERS HANDLER ------------------------
------------------------------------------------------------

local worker_methods,worker_class_metatable = class("MapReduceCommon.worker")

function worker_class_metatable:__call(name,address,port)
  local obj = { name=name, address=address, port=port }
  return class_instance(obj,self,true)
end

local function check_error(p,msg)
  if not p then fprintf(io.stderr, "ERROR: %s\n", msg) end
  return p
end

function worker_methods:ping(select_handler, timeout)
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  s:settimeout(timeout)
  if not check_error(s,error_msg) then return false end
  ok,error_msg=s:connect(self.address,self.port)
  if not check_error(ok,error_msg) then return false end
  --
  select_handler:send(s, "PING")
  select_handler:receive(s,
			 function(conn, msg)
			   if msg ~= "PONG" then
			     self:set_dead()
			   end
			 end)
  select_handler:send(s, "EXIT")
  select_handler:receive(s)
  select_handler:close(s, function() end)
end

function worker_methods:update(address,port)
  self.address = address
  self.port    = port
end

function worker_methods:get()
  return self.name,self.address,self.port
end

function worker_methods:set_dead()
  self.name = nil
  self.address = nil
  self.port = nil
end

function worker_methods:dead()
  return not self.name and not self.address and not self.port
end

