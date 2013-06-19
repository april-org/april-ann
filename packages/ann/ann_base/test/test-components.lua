-----------------
-- DOT PRODUCT --
-----------------
w = matrix(3, 2, {1, 2,
		  3, 4,
		  5, 6})
c = ann.components.dot_product{ input=2, output=3, weights="w" }
c:build{ weights = { w = ann.connections{ input=2, output=3, w = w } } }
i = matrix(2, 2, {5, 6,
		  7, 8})
o = c:forward(tokens.matrix( i:clone("col_major") )):get_matrix()

if o ~= (i*w:transpose()) then
  print(o)
  print(i*w:transpose())
  error("Error!!!")
end

j = matrix(2, 3, {1, 2, 3,
		  4, 5, 6})
o = c:backprop(tokens.matrix( j:clone("col_major") )):get_matrix()
if o ~= (j*w)then
  print(o)
  print(j*w)
  error("Error!!!")
end

i = matrix(1, 2, {10, 15})
o = c:forward(tokens.matrix( i:clone("col_major") )):get_matrix()

if o ~= (i*w:transpose()) then
  print(o)
  print(i*w:transpose())
  error("Error!!!")
end

j = matrix(1, 3, {10, 15, 20})
o = c:backprop(tokens.matrix( j:clone("col_major") )):get_matrix()
if o ~= (j*w)then
  print(o)
  print(j*w)
  error("Error!!!")
end

----------
-- BIAS --
----------

b = matrix(4, {1, 2, 3, 4})
c = ann.components.bias{ size=4, weights="b" }
c:build{ weights = { b = ann.connections{ input=1, output=4, w = b } } }
i = matrix(4, {5, 6, 7, 8})
o = c:forward(tokens.matrix( i:clone("col_major") )):get_matrix()
if o ~= (b+i) then
  print(o)
  print(b+i)
  error("Error!!!")
end
