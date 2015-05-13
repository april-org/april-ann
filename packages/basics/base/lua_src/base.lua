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

local function lookup(obj, obj_id, cls)

end

setmetatable(cast.to, {
               __call = function(self, obj, cls)
                 local obj_id = type(obj)
                 local meta = assert(cls.meta_instance,
                                     "Needs a target class as 2nd argument")
                 local f
                 while not f do
                   local cls_id = assert(meta.id)
                   if cls_id == obj_id then return obj end
                   local to = cast.to[obj_id]
                   if to then f = to[cls_id] end
                   if not f then
                     local meta2 = assert(getmetatable(meta.__index),
                                          "Incorrect derived class")
                     print(meta, meta2)
                     if raweq(meta, meta2) then
                       assert("Unable casting to given class")
                     end
                     meta = meta2
                   end
                 end
                 return f(obj)
               end
})
