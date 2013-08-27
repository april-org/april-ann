-- gzio is used exactly the same way as standard module io

f=gzio.open("test.gz", "w")
f:write("Hello, World.\n")
f:close()

f=io.open("test.gz")
assert(f:read("*l"), "Hello, World.")
f:close()

for line in io.lines("test.gz") do assert(line, "Hello, World.") end

os.remove("test.gz")

-- the tar module reads tar files, can be used with gzio in order to read tgz
f=gzio.open(string.get_path(arg[0]).."a.tar.gz")
t=tar.open(f)

data = {
  ["dos.txt"] = {
    "Hola, soy dos.txt.",
    "Tengo mas lineas que uno.txt.",
    "Pero no muchas mas.",
    "Adios.",
  },
  ["uno.txt"] = {
    "Hola, soy uno.txt.",
  },
}

for i in t:files() do
  assert(data[i], string.format("File %s is not expected",i))
  f2=t:open(i)
  local all = f2:read("*a")
  assert(all == table.concat(data[i],"\n"),
	 string.format("Problem with file %s, expected\n%s\nFound\n%s\n",
		       i, table.concat(data[i],"\n"), all))
  f2:close()
end
