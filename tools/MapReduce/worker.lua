require "socket"
require "common"
-- module worker
worker = {}

local function execute(core)
  local logger = core.logger
  local mmap,mreduce,share = core.map,core.reduce,core.share
  while true do
    collectgarbage("collect")
    local msg = core:read()
    local action,data = msg:match("^%s*([^%s]+)%s*(.*)")
    print(action)
    
    if action == "CLOSE" then core:close() break

    elseif action == "MAP" then
      local map_key,value = data:match("^%s*([^%s]*)%s*(.*)")
      value = common.load(value,logger)
      -- TODO: check value error
      local map_result = mmap(map_key,value)
      core:printf("return %s",table.tostring(map_result))
      
    elseif action == "REDUCE" then
      local key,values = data:match("^%s*([^%s]*)%s*(.*)")
      values = common.load(values,logger)
      -- TODO: check reduce values error
      local key,result = mreduce(key,values)
      core:printf("return %s %s",key,result)
      
    elseif action == "SHARE" then
      share(data)
    end
  end
  os.exit(0)
end

-- The core class is an abstraction layer which forks the process using two
-- pipes for bidirectional communication, so the parent and the child process
-- communicate by menas of this pipes. The class has do_map and do_reduce
-- methods, and when it is busy (doing computation), a temporal file is created,
-- and when it is free, the temporal file is removed, so the methods lock(),
-- unlock() and busy() use the temporal file to control the state of the object.
local core_methods,core_class_metatable = class("worker.core")

function core_class_metatable:__call(logger,tmpname,script,arg,taskid)
  local tochild  = table.pack(util.pipe())
  local toparent = table.pack(util.pipe())
  local obj = { tmpname=tmpname, logger=logger, taskid=taskid }
  local arg = common.load(arg)
  local t = common.load(script,logger,table.unpack(arg))
  -- TODO: check script error
  obj.map    = t.map
  obj.reduce = t.reduce
  obj.share  = t.share or function() end
  --
  who,obj.pid = util.split_process(2)
  if obj.pid then
    -- the parent
    obj.IN  = toparent[1]
    obj.OUT = tochild[2]
    obj = class_instance(obj,self)
    obj:unlock()
  else
    -- the child
    signal.release(signal.SIGINT)
    obj.IN  = tochild[1]
    obj.OUT = toparent[2]
    obj = class_instance(obj,self)
    execute(obj)
    os.exit(0)
  end
  return obj
end

-- static method
function worker.core.wait()
  util.wait()
end

-- function core_methods:

function worker.core.meta_instance:__gc()
  if self.pid then
    self:close()
  end
end

function core_methods:close()
  if self.pid then
    self:write("CLOSE NOW\n")
    self:flush()
  end
  self.OUT:close()
  self.IN:close()
  if self.pid then
    util.wait()
  end
end

function core_methods:flush()
  self.OUT:flush()
  self.IN:flush()
end

function core_methods:write(msg)
  local len = binarizer.code.uint32(#msg)
  self.OUT:write(len)
  self.OUT:write(msg)
  self.OUT:flush()
end

function core_methods:print(...)
  self:write(table.concat(...,"\t").."\n")
end

function core_methods:printf(format,...)
  self:write(string.format(format,...))
end

function core_methods:read()
  self.IN:flush()
  local len = binarizer.decode.uint32(self.IN:read(5))
  local msg = self.IN:read(len)
  return msg
end

function core_methods:lock(msg)
  local f = io.open(self.tmpname,"w")
  f:write(msg)
  f:close()
end

function core_methods:unlock()
  os.remove(self.tmpname)
end

function core_methods:busy()
  local f = io.open(self.tmpname)
  if f then f:close() return true end
  return false
end

function core_methods:do_map(master_address,master_port,
			     map_key,value,
			     pending_pids)
  self:lock("MAP\n")
  if self.pid then
    local _,pid = util.split_process(2)
    if pid then
      -- parent
      table.insert(pending_pids, pid)
    else
      -- child
      signal.release(signal.SIGINT)
      local taskid = self.taskid
      self:printf("MAP %s %s\n", map_key, value)
      local result = self:read()
      -- CONNECTION WITH MASTER
      local ok,error_msg,conn,data = true
      conn,error_msg = socket.tcp()
      -- TODO: check error
      -- if not check_error(s,error_msg) then return false end
      ok,error_msg=conn:connect(master_address,master_port)
      -- TODO: check error
      --if not check_error(ok,error_msg) then return false end
      common.send_wrapper(conn,
			  string.format("MAP_RESULT %s %s %s",
					taskid, map_key, result))
      local ok = common.recv_wrapper(conn)
      -- TODO: throw error if ok~="EXIT"
      common.send_wrapper(conn, "EXIT")
      conn:close()
      self:unlock()
      os.exit(0)
    end
  end
end

-- remember to mark as dead the given connection
function core_methods:do_reduce(master_address,master_port,
				key,values,
				pending_pids)
  self:lock("REDUCE\n")
  if self.pid then
    local _,pid = util.split_process(2)
    if pid then
      -- parent
      table.insert(pending_pids, pid)
    else
      -- child
      signal.release(signal.SIGINT)
      local taskid = self.taskid
      self:printf("REDUCE %s %s\n", key, values)
      local msg = self:read()
      local key,value = msg:match("^%s*([^%s]+)%s*(.*)$")
      -- CONNECTION WITH MASTER
      local ok,error_msg,conn,data = true
      conn,error_msg = socket.tcp()
      -- TODO: check error
      -- if not check_error(s,error_msg) then return false end
      ok,error_msg=conn:connect(master_address,master_port)
      -- TODO: check error
      --if not check_error(ok,error_msg) then return false end
      common.send_wrapper(conn,
			  string.format("REDUCE_RESULT %s %s %s",
					taskid, key, value))
      local ok = common.recv_wrapper(conn)
      -- TODO: throw error if ok~="EXIT"
      common.send_wrapper(conn, "EXIT")
      conn:close()
      self:unlock()
      os.exit(0)
    end
  end
end

function core_methods:share(data)
  if self.pid then
    self:printf("SHARE %s",data)
  end
end
