tabladice = {0.4,0.2,0.1,0.1,0.1,0.1}
dado = random.dice(tabladice)
aleat = random()
histogram = {}
for i=1,dado:outcomes() do table.insert(histogram,0) end
veces = 10000
for i=1,veces do
  resul = dado:thrown(aleat)
  --printf("%d,",resul)
  histogram[resul] = histogram[resul]+1
end

print"---------------------------------------"
for i=1,dado:outcomes() do
  printf("Histograma[%d] = %.3f should be %.3f\n",
	 i,histogram[i]/veces,tabladice[i])
end

