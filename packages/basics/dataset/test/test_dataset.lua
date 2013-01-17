print("Empieza el programa")

m = matrix(10,{1,0,0,0,0,0,0,0,0,0})

print(m)

ds = dataset.matrix(m,{patternSize={10}, offset={0}, numSteps={20},
		      stepSize={-1}, circular = {true}})

print(ds)

-- for index,pattern in ds:patterns() do
--   print(index,"{"..table.concat(pattern,",").."}")
-- end

print("Termina el programa")

