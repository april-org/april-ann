
lineas = [[
a * b b b b
a a a * b
a b c * b
b * a b c
a b c * a a b b
esto es una prueba*esto es prueba
de un fichero de palabras*de un fichero de muchas palabras
para probar tasas*para probar el programa tasas
]]

lineas = string.tokenize(lineas,"\n")
for _,line in ipairs(lineas) do
  print("->"..line)
end

resul = tasas{
  datatype = "lines", -- no hace falta, es valor por defecto
  line_sep = "%*",    -- no hace falta, es valor por defecto
  words_width = 1,
  --  words_sep = " ",    -- no hace falta, por defecto es ' \t'
  data = lineas, -- los datos
  tasa = "pra", -- campo obligatorio, no hay valor por defecto
  confusion_matrix=true, -- por defecto es false
}

for i,j in pairs(resul) do
  print(i,j)
end

print"\n------ matriz de confusion -------"
for i,j in pairs(resul.confusion_matrix) do
  for k,l in pairs(j) do
    printf("%-8s x %-8s -> %2d ('%s' '%s')\n",i,k,l,i,k)
  end
end
print"----------------------------------\n"

print"SIN MATRIZ DE CONFUSION"

resul = tasas{
  tasa = "pra",
  data = lineas,
}

for i,j in pairs(resul) do
  print(i,j)
end

------------------------------------------------

print"CON ENTEROS"

enteros = {
  {{1,1,2,3},{1,3,2,2,4}},
  {{1,1,2,3},{1,1,2,3,4}}
}
resul = tasas{
  tasa = "pra",
  data = enteros,
  datatype = "pairs_int",
  confusion_matrix=true,
}

for i,j in pairs(resul) do
  print(i,j)
end

print"matriz de confusion:"
for i,j in pairs(resul.confusion_matrix) do
  for k,l in pairs(j) do
    printf("%d x %d -> %d\n",i,k,l)
  end
end

------------------------------------------------

print"\n-----------------------------------------"
print"MODO RAW"

enteros = {
  {{1,1,2,3},{1,3,2,2,4}},
  {{1,1,2,3},{1,1,2,3,4}}
}
resul = tasas{
  tasa = "raw",
  data = enteros,
  datatype = "pairs_int",
}

for i,j in ipairs(resul) do
  for k,l in pairs(j) do
    print(i,k,l)
  end
end

