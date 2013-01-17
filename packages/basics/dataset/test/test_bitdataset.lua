numPatterns = 100
patternSize = 100

m     = matrix(numPatterns, patternSize)
ds    = dataset.matrix(m)
dsbit = dataset.bit{
  patternSize = patternSize,
  numPatterns = numPatterns
}

rnd = random(1234)

for i=1,numPatterns do
  t = {}
  for j=1,patternSize do
    v = rnd:randInt(0,1)
    t[j] = v
  end
  ds:putPattern(i, t)
  dsbit:putPattern(i, t)
  --  print(table.concat(t, " "))
end

for i=1,numPatterns do
  local t1 = ds:getPattern(i)
  local t2 = dsbit:getPattern(i)
  for j=1,#t1 do
    if t1[j] ~= t2[j] then
      print(i, j, t1[j], t2[j])
      error("ERROR")
    end
  end
end

print("Test1 OK!")

---------------------------------------------------

dsbit = dataset.bit{
  dataset = ds
}

for i=1,numPatterns do
  local t1 = ds:getPattern(i)
  local t2 = dsbit:getPattern(i)
  for j=1,#t1 do
    if t1[j] ~= t2[j] then
      print(i, j, t1[j], t2[j])
      error("ERROR")
    end
  end
end

print("Test2 OK!")
