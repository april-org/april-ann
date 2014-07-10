result = { 1, 1, 1, 5, 5, 5 }

mfset = util.mfset()

mfset:merge(1,2)
mfset:merge(2,3)
mfset:merge(4,6)
mfset:merge(5,6)
mfset:print()
for i=1,mfset:size() do
  if mfset:find(i) ~= result[i] then
    error ("Error en el merge o en el find!!!")
  end
  print (i, mfset:find(i))
end

print("==========================")

t=util.mfset.fromString(mfset:toString())
t:print()
for i=1,t:size() do
  if t:find(i) ~= mfset:find(i) then
    error ("Error en el metodo toString o fromString!!!")
  end
  print (i, t:find(i))
end
