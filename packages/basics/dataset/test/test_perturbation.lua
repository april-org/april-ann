d = dataset.identity(5)

print("Dataset sin perturbar:")
for i,j in d:patterns() do
  print(table.concat(j,","))
end

p = dataset.perturbation{
  dataset   = d,
  random    = random(123),
  mean      = 0,   -- de la gaussiana
  variance  = 0.1, -- de la gaussiana
}

for veces = 1,4 do
  print("Dataset perturbado:")
  for i,j in p:patterns() do
    print(table.concat(j,","))
  end
end



