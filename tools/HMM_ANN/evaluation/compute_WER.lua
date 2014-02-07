if #arg ~= 2 then
  printf("SINTASIS: %s REF 1BEST\n", string.basename(arg[0]))
  os.exit(1)
end
local rf = io.open(arg[1])
local target = io.open(arg[2])

SER = 0
n = 0
rates_list = {}
for correcta in rf:lines() do
  local reconocida = target:read("*l")
  if not reconocida then
    print("Warning!!! target shorter than reference")
    break
  end
  if reconocida ~= correcta then SER = SER + 1 end
  n = n + 1
  table.insert(rates_list, { correcta, reconocida })
end
resul = rates{
  datatype = "pairs_lines",
  data = rates_list,
  rate = "ie",
}
resul_cer = rates{
  datatype = "pairs_lines",
  data = rates_list,
  rate = "ie",
  words_width = 1,
  confusion_matrix = true,
}

tot = resul.borr + resul.sust + resul.ac
printf ("WER: %f %%  i:%f %%   d:%f %%   s:%f %%   a:%f %%\n",
	resul.rate, resul.ins/tot*100, resul.borr/tot*100,
	resul.sust/tot*100, resul.ac/tot*100)
printf ("CER: %f %%\n", resul_cer.rate)
printf ("SER: %f %%\n", SER/n*100)

local f = io.open("confus.matrix", "w")
fprintf(f, "------ matriz de confusion -------\n")
for i,j in pairs(resul_cer.confusion_matrix) do
  for k,l in pairs(j) do
    fprintf(f, "%-8s x %-8s -> %2d (%s)\n",i,k,l,
	    (i==k and "DIAG") or "NOT-DIAG")
  end
end
fprintf(f, "----------------------------------\n")
f:close()
