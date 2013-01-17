-- gzio is used exactly the same way as standard module io

f=gzio.open("test.gz", "w")
f:write("Hello, World.\n")
f:close()

-- the tar module reads tar files, can be used with gzio in order to read tgz
f=gzio.open("a.tar.gz")
t=tar.open(f)

for i in t:files() do
    print("------", i)
    f2=t:open(i)
    print(f2:read("*a"))
    f2:close()
end

