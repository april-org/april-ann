require "socket"
require "common"
-- module worker
worker = {}

local function execute(core)
  local logger = core.logger
  local mmap,mreduce,share = core.map,core.reduce,core.share
  local N = 0
  while true do
    N = N + 1 if N > 100 then collectgarbage("collect") N = 0 end
    local msg = core:read()
    local action,data = msg:match("^%s*([^%s]+)%s*(.*)")
    
    if action == "CLOSE" then core:close() break
      
    elseif action == "MAP" then
      local str = data:match("^%s*(return .*)")
      local map_key,value = common.load(str,logger)
      -- TODO: check value error
      local map_result = mmap(map_key,value)
      map_result = common.tostring(map_result)
      core:open_result_file()
      core:write_result(string.format("MAP_RESULT {return %s} return %s",
				      common.tostring(map_key,true),
				      map_result))
      core:close_result_file()
      --
      core:unlock()
      core:wakeup_worker()
      
    elseif action == "REDUCE" then
      core:open_result_file()
      local str = data:match('^%s*(return .*)')
      local key,values = common.load(str,logger)
      -- TODO: check reduce values error
      local key,result = mreduce(key,values)
      key=common.tostring(key, true)
      result=common.tostring(result)
      core:write_result(string.format("REDUCE_RESULT return %s,%s",key,result))
      
    elseif action == "REDUCE_READY" then
      core:close_result_file()
      core:unlock()
      core:wakeup_worker()
      
    elseif action == "SHARE" then
      local data = common.load(data)
      -- TODO: check error
      share(data)
      --
      core:unlock()
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

function core_class_metatable:__call(logger,tmpname,script,arg,taskid,
				     workersock,connections,
				     port)
  local tochild  = table.pack(util.pipe())
  local obj = {
    tmpname=tmpname,
    result_tmpname=tmpname.."_result",
    logger=logger,
    taskid=taskid,
    port=port,
    result_f="closed"
  }
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
    obj.OUT = tochild[2]
    obj = class_instance(obj,self)
    obj:unlock()
  else
    -- the child
    workersock:close()
    connections:close()
    -- signal.release(signal.SIGINT)
    obj.IN  = tochild[1]
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

function core_methods:wakeup_worker()
  if not self.pid then
    local conn = common.connections_set.connect("localhost",self.port)
    conn:close()
  end
end

function core_methods:close()
  if self.pid then
    self:write("CLOSE NOW\n")
    self:flush()
  end
  if self.OUT then self.OUT:close() end
  if self.IN  then self.IN:close()  end
  if self.pid then
    util.wait()
  end
end

function core_methods:flush()
  if self.OUT then self.OUT:flush() end
  if self.IN  then self.IN:flush()  end
end

function core_methods:write(msg)
  if self.OUT then
    local len = binarizer.code.uint32(#msg)
    self.OUT:write(len)
    self.OUT:write(msg)
    self.OUT:flush()
  end
end

function core_methods:print(...)
  self:write(table.concat(...,"\t").."\n")
end

function core_methods:printf(format,...)
  self:write(string.format(format,...))
end

function core_methods:read()
  if self.IN then
    self.IN:flush()
    local len = binarizer.decode.uint32(self.IN:read(5))
    local msg = self.IN:read(len)
    return msg
  end
end

function core_methods:lock(msg)
  local f = io.open(self.tmpname,"w")
  f:write(msg)
  f:close()
end

function core_methods:unlock()
  -- FIXME: Is not working, why??? os.remove(self.tmpname)
  os.execute("rm -f " .. self.tmpname)
end

function core_methods:busy()
  local f = io.open(self.tmpname)
  if f then f:close() return true end
  return false
end

function core_methods:read_pending()
  local f = io.open(self.result_tmpname)
  if f then f:close() return true end
  return false
end

function core_methods:open_result_file()
  if not self.pid then
    if self.result_f == "closed" then
      self.result_f = io.open(self.result_tmpname, "w")
    end
  end
end

function core_methods:write_result(result)
  if not self.pid then
    local f = self.result_f
    local len = binarizer.code.uint32(#result)
    f:write(len)
    f:write(result)
  end
end

function core_methods:close_result_file()
  if not self.pid then
    self.result_f:close()
    self.result_f = "closed"
  end
end

function core_methods:read_result()
  if self.pid then
    local f = io.open(self.result_tmpname)
    if not f then return nil end
    local result = f:read("*a")
    f:close()
    -- FIXME: It is not working... why???? print(os.remove(self.result_tmpname))
    os.execute("rm -f " .. self.result_tmpname)
    return result
  end
end

function core_methods:do_map(map_key_and_value)
  if self.pid then
    self:lock("MAP\n")
    self:printf("MAP %s\n", map_key_and_value)
  end
end

-- remember to mark as dead the given connection
function core_methods:begin_reduce_bunch()
  if self.pid then
    self:lock("REDUCE\n")
  end
end

function core_methods:append_reduce_bunch(key_and_value)
  if self.pid then
    self:printf("REDUCE %s\n", key_and_value)
  end
end

function core_methods:end_reduce_bunch()
  if self.pid then
    self:printf("REDUCE_READY")
  end
end

function core_methods:share(data)
  if self.pid then
    self:lock("SHARE")
    self:printf("SHARE %s",data)
  end
end
