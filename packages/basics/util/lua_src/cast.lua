cast = cast or {}

local registry = debug.getregistry()
registry.luabind_cast = registry.luabind_cast or {}
local luabind_cast = registry.luabind_cast
assert(type(luabind_cast) == "table",
       "Registry 'luabind_cast' field should be a table")
       
local function try_cast(obj, cls_id)
  local obj_id = type(obj)
  assert(obj_id ~= cls_id)
  local tbl = luabind_cast[obj_id]
  if tbl then
    local f = tbl[cls_id]
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

assert(not cast.to, "cast.to has been set, it cannot be overwritten")
cast.to = function(obj, cls)
  assert(class.is_class(cls), "Needs a class as 2nd argument")
  local metainst = assert(cls.meta_instance,
                          "Needs a target class as 2nd argument")
  return lookup(obj, metainst)
end
