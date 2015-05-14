cast = cast or {}
cast.to = cast.to or {}

local function try_cast(obj, cls_id)
  local obj_id = type(obj)
  assert(obj_id ~= cls_id)
  local to = cast.to[obj_id]
  if to then
    local f = to[cls_id]
    if f then return f(obj) end
  end
  return nil,"Unable to locate cast function"
end

local function lookup(obj, metainst)
  local obj_id = type(obj)
  local cls_id = assert(metainst.id, "Unable to locate 'id' field")
  -- base case
  if obj_id == cls_id then return obj end
  -- general case, traverse cls table recursively
  local metainst2 = getmetatable(metainst.__index)
  if not metainst2 then return nil,"Incorrect derived class" end
  if rawequal(metainst, metainst2) then
    return nil,"Unable casting to given class"
  end
  local obj,msg = lookup(obj, metainst2)
  if obj then obj,msg = try_cast(obj, cls_id) end
  return obj,msg
end

setmetatable(cast.to, {
               __call = function(self, obj, cls)
                 assert(class.is_class(cls), "Needs a class as 2nd argument")
                 local metainst = assert(cls.meta_instance,
                                         "Needs a target class as 2nd argument")
                 return lookup(obj, metainst)
               end
})
