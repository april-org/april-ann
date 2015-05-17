do
  local a = aprilio.stream.file("AUTHORS.txt")
  local b = cast.to(a, aprilio.stream.file)

  local a = nil
  local b = nil

  local t = {}
  for i=1,1000 do
    t[i] = i
  end
  collectgarbage("collect")
end
collectgarbage("collect")

april_list(debug.getregistry().luabind_refs)
