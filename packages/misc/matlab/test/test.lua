dir  = string.get_path(arg[0])
test1 = matlab.read(dir.."test1.mat")
test2 = matlab.read(dir.."test2.mat")
test3 = matlab.read(dir.."test3.mat")
test4 = matlab.read(dir.."test4.mat")

print(test1.x)

print(test2.C:get(1,1))
print(test2.C:get(1,2))

print(test3.X.y)
print(test3.X.w)
print(test3.X.z)

print(test4.A)
