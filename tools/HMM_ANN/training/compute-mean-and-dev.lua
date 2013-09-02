if #arg ~= 1 then
  print("SINTAXIS: %s file_list\n"..
	"file_list must contains filenames of "..
	"matrix parameters (April format)\n",
	arg[0])
  os.exit(0)
end

lista = arg[1]

local ds = {}
for fichero in io.lines(lista) do
  fprintf(io.stderr, "%s\n", fichero)
  local f   = io.open(fichero) or error ("File not found: " .. fichero)
  local mat = matrix.fromFilename(fichero)
  table.insert(ds, dataset.matrix(mat))
end
ds         = dataset.union(ds)
means,devs = ds:mean_deviation()
printf("-- %d patterns, %d params\n", ds:numPatterns(), ds:patternSize())
printf("return {\nmeans = {\n%s\n},\ndevs = {\n%s\n}\n}\n",
       table.concat(means, ", "),
       table.concat(devs, ", "))
