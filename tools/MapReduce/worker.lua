require "socket"
require "common"
-- module worker
worker = {}

local function execute(core)
  local logger = core.logger
  local map,reduce
  while true do
    collectgarbage("collect")
    local msg = core:read()
    local action,data = msg:match("([^%s]+) (.*)")
    print(action)

    if action == "CLOSE" then core:close() break

    elseif action == "MAP" then
      local map_key,job = data:match("([^%s]*) (.*)")
      job = common.load(logger,job)
      -- TODO: check job error
      local map_result = map(map_key,job)
      self:write(table.tostring(map_result))

    elseif action == "REDUCE" then
      local key,values = data:match("([^%s]*) (.*)")
      values = common.load(logger,values)
      -- TODO: check reduce values error
      local key,result = reduce(key,values)
      self:write(key)
      self:write(result)
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

function core_class_metatable:__call(logger,tmpname,script,taskid)
  local tochild  = table.pack(util.pipe())
  local toparent = table.pack(util.pipe())
  local obj,who = { tmpname=tmpname, logger=logger, taskid=taskid }
  local t = common.load(logger,script)
  -- TODO: check script error
  map    = t.map
  reduce = t.reduce
  self:unlock()
  --
  who,obj.pid = util.split_process(2)
  if obj.pid then
    -- the parent
    obj.IN  = toparent[1]
    obj.OUT = tochild[2]
    return class_instance(obj,self)
  else
    -- the child
    obj.IN  = tochild[1]
    obj.OUT = toparent[2]
    obj = class_instance(obj,self)
    execute(obj)
  end
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
end

function core_methods:print(...)
  self.OUT:write(table.concat(...,"\t").."\n")
end

function core_methods:printf(format,...)
  self:write(string.format(format,...))
end

function core_methods:read()
  local len = binarizer.decode.uint32(self.IN:read(5))
  return self.IN:read(len)
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

-- remember to mark as dead the given connection
function core_methods:do_map(conn,select_handler,map_key,job)
  self:lock("MAP\n")
  if self.pid then
    local _,pid = util.split_process(2)
    if pid then
      -- parent
      select_handler:close(conn)
    else
      -- child
      local taskid = self.taskid
      self:printf("MAP %s %s\n", map_key, job)
      local result = self:read()
      common.send_wrapper(conn,
			  string.format("%s %s %s",
					taskid, map_key, result))
      conn:close()	
      self:unlock()
      os.exit(0)
    end
  end
end

-- remember to mark as dead the given connection
function core_methods:do_reduce(conn,select_handler,key,values)
  self:lock("REDUCE\n")
  if self.pid then
    local _,pid = util.split_process(2)
    if pid then
      -- parent
      select_handler:close(conn)
    else
      -- child
      local taskid = self.taskid
      self:printf("REDUCE %s %s\n", key, values)
      local key,result = self:read(),self:read()
      common.send_wrapper(conn,
			  string.format("%s %s %s",
					taskid, key, result))
      conn:close()
      self:unlock()
      os.exit(0)
    end
  end
end
