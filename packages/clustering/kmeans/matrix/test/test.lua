require "kmeans_matrix"

filename = arg[1] or 'data.txt'
K = arg[2] or 2
data = matrix.fromFilename(filename)

clusters = data({1,K},':'):clone()

print(clusters)

clock = util.stopwatch()
clock:go()

res = clustering.kmeans.matrix{ data=data, centroids=clusters }

clock:stop()
cpu, wall = clock:read()
print(clusters)
print(res)
print(cpu, wall)
