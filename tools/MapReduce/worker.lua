require "socket"
-- module worker
worker = {}

local function execute(core)
  while true do
    local msg = core:read()
    print(msg)
    if msg == "CLOSE\n" then core:close() break end
  end
end

local core_methods,core_class_metatable = class("worker.core")

function core_class_metatable:__call()
  local tochild  = table.pack(util.pipe())
  local toparent = table.pack(util.pipe())
  local obj,who = { }
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

function core_methods:close()
  self:write("CLOSE\n")
  self:flush()
  self.OUT:close()
  self.IN:close()
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

function core_methods:do_map(conn,select_handler,map_key,job)
  if self.pid then
    local _,pid = util.split_process(2)
    if pid == 0 then
      self:printf("MAP %s %s\n", map_key, job)
    else
      select_handler:close(conn)
    end
  end
end
