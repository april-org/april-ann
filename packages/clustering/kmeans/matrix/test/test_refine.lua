require "kmeans_matrix"


filename = arg[1] or 'data.txt'
data = matrix.fromFilename(filename)


K = 2


res = clustering.kmeans.matrix({ 
    data = data,
    K = K,
    random = random(1234)
})

print(res)
