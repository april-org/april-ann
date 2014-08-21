local check = utest.check
local T = utest.test

T("GZIPTest", function()
    -- gzio is used exactly the same way as standard module io
    
    local f=gzio.open("test.gz", "w")
    f:write("Hello, World.\n")
    f:close()
    
    local f=gzio.open("test.gz")
    check.eq(f:read("*l"), "Hello, World.")
    f:close()

    for line in gzio.lines("test.gz") do check.eq(line, "Hello, World.") end
    os.remove("test.gz")
    
end)

T("TARGZTest", function()
    -- the tar module reads tar files, can be used with gzio in order to read tgz
    local f = gzio.open(string.get_path(arg[0]).."a.tar.gz")
    local t = tar.open(f)
    
    local data = {
      ["dos.txt"] = {
        "Hola, soy dos.txt.",
        "Tengo mas lineas que uno.txt.",
        "Pero no muchas mas.",
        "Adios.",
        "",
      },
      ["uno.txt"] = {
        "Hola, soy uno.txt.",
        "",
      },
    }
    
    check.eq(t:number_of_files(), 2)
    for i in t:files() do
      check.TRUE(data[i], string.format("File %s is not expected",i))
      local f2=t:open(i)
      local all = f2:read("*a")
      check.eq(all, table.concat(data[i],"\n"),
               string.format("Problem with file %s, expected\n%s\nFound\n%s\n",
                             i, table.concat(data[i],"\n"), all))
      f2:close()
    end
end)
