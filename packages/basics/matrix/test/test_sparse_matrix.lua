local blockf = mathcore.block.float
local blocki = mathcore.block.int32
local a = matrix.sparse.csc(10,10,
			    blockf{1,1,1,1,1,1,1,1,1,1},
			    blocki{0,1,2,3,4,5,6,7,8,9},
			    blocki{0,1,2,3,7,7,7,7,8,9,10})
iterator(a:iterate()):apply(print)

print(a)

local d   = matrix(4,4):zeros():diag(1)
local aux = matrix.sparse.csr(d)
print(aux)

local aux2 = matrix.sparse.diag(matrix(5,{1,2,3,4,5}),"csc")
print(aux2)

local aux3 = matrix.sparse.diag(blockf({1,2,3,4,5,4,3,2,1}),"csc")
print(aux3)

print(2*aux3)

print(aux3:max())

b = aux3('3:6','4:6')
print(b)

local m = matrix(9,1):linspace()
local m2 = matrix.sparse.csr(3,3,
			     blockf{10,-4},
			     blocki{1,0},
			     blocki{0,1,2,2})
local aux = m2:as_vector()
print(m:clone():axpy(1.0,aux))

local b = matrix.sparse.csr(1,3,blockf{3},blocki{0})
print(b)

