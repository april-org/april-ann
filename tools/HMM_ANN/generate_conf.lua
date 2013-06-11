conf=arg[1]

optargs = {}
if io.open(conf) then
  local t = dofile(conf)
  for name,value in pairs(t) do
    optargs[name] = value
  end
  i=2
else
  i=1
end

while i <= #arg do
  local t = {}
  if string.sub(arg[i], 1, 2) == "--" then
    local r = string.tokenize(arg[i], "=")
    if r[1] ~= "--defopt" then
      t[1] = string.gsub(r[1], "%-%-", "")
      t[1] = string.gsub(t[1], "%-", "_")
      t[2] = r[2]
    end
  else
    t[1] = string.sub(arg[i], 2, 2)
    if t[1] ~= "f" then
      if string.sub(arg[i+1] or "-", 1, 1) == "-" then
	t[2] = string.sub(arg[i], 3, #arg[i])
      else
	i=i+1
	t[2] = arg[i]
      end
    end
  end
  if #t == 2 then
    optargs[t[1]] = t[2]
  end
  i=i+1
end

print("return {")
for name,value in pairs(optargs) do
  printf("  %s=", name)
  if tonumber(value) then
    printf ("%g,\n", tonumber(value))
  else
    printf ("\"%s\",\n", value)
  end
end
print("}")
