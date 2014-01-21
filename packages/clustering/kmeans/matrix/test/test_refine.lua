filename = arg[1] or 'data.txt'
data = matrix.fromFilename(filename)


K = 2


res,C = clustering.kmeans.matrix({ 
    data = data,
    K = K,
    random = random(1234)
})

print(C)
print(res)
