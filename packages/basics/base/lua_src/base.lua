aprilann = { _NAME = "APRIL-ANN" }

make_deprecated_function = function(name, new_name, new_func)
  return function(...)
    if new_func then
      if new_name then
        io.stderr:write(debug.traceback(string.format("Warning: %s is in deprecated state, use %s instead",
                                                      name, new_name)))
      else
        io.stderr:write(debug.traceback(string.format("Warning: %s is in deprecated state",
                                                      name)))
      end
      io.stderr:write("\n")
      return new_func(...)
    else
      error(string.format("%s is in deprecated state%s", name,
                          new_name and (", currently it is %s"%{new_name}) or ""))
    end
  end
end

cast = cast or {}
cast.to = cast.to or {}

local function lookup(obj_id, cls)
  local meta = assert(cls.meta_instance,
                      "Needs a target class as 2nd argument")
  while true do
    local cls_id = assert(meta.id)
    local to = cast.to[obj_id]
    if to then return to[cls_id] end
    local meta2 = assert(getmetatable(meta.__index), "Incorrect derived class")
    if raweq(meta, meta2) then break end
    meta = meta2
  end
end

setmetatable(cast.to, {
               __call = function(self, obj, cls)
                 local obj_id = type(obj)
                 local f = assert(lookup(obj_id, cls),
                                  "Unable casting to given class")
                 return f(obj)
               end
})
