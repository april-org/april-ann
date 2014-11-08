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
