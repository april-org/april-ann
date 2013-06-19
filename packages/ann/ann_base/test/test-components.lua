w = matrix(2, 2, {1, 2, 3, 4})
c = ann.components.dot_product{ input=2, output=2, weights="w" }
c:build{ weights = { w = ann.connections{ input=2, output=2, w = w } } }
i = matrix(2, 2, {5, 5, 5, 5})
o = c:forward(tokens.matrix( i:clone("col_major") )):get_matrix()
print(o)
print(w*i)

b = matrix(4, {1, 2, 3, 4})
c = ann.components.stack():push(ann.components.base{ size=4 }):push(ann.components.bias{ weights="b" })
c:build{ input=4, output=4, weights = { b = ann.connections{ input=1, output=2, w = b } } }
i = matrix(4, {5, 5, 5, 5})
o = c:forward(tokens.matrix( i:clone("col_major") )):get_matrix()
print(o)
print(w+i)
