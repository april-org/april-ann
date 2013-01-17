print("Empieza el programa")

a = random()
b = a:clone()
for i=1,10 do
  print("randInt 1 a 10",a:randInt(1,10))
end
print("-----------------------------")
print("Debe salir la misma secuencia:")
for i=1,10 do
  print("randInt 1 a 10",b:randInt(1,10))
end

numpermutaciones = 10
tallapermutacion = 10
printf("Vamos a ver %d permutaciones de %d elementos\n",
     numpermutaciones,
     tallapermutacion)
for i=1,numpermutaciones do
  print(string.join(a:shuffle(tallapermutacion),","))
end

print("-----------------------------")
print("VAMOS A PERMUTAR UNA TABLA")

tabla = {"uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho" }

print("Tabla original : "..table.concat(tabla,","))
for i=1,numpermutaciones do
  print("Tabla permutada: "..table.concat(a:shuffle(tabla),","))
end


print("Termina el programa")

