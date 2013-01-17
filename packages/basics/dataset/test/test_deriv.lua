d = dataset.identity(10)

print("Dataset original:")
for i,j in d:patterns() do
  print(table.concat(j,","))
end

p = dataset.deriv{
  dataset = d,
  deriv0  = false;
  deriv1  = true; -- default value
  deriv2  = false;
}

print("Dataset con derivadas y aceleraciones:")
for i,j in p:patterns() do
  print(table.concat(j,","))
end

