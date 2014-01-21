require "kmeans_matrix"

x = {} 
for i=1,5 do table.insert(x,i) table.insert(x,i) end
data = matrix(5,2,x)

clusters = matrix(2,2,{ 0,0,
			1,1,
		      })
K = 2

res, C = clustering.kmeans.matrix({ 
    data = data,
    K = K,
    centroids = clusters
    
})

print(clusters)
print("Distorsion:",res)

points = matrix(2,2,{ 0,0,
		      10,10,
		    })
tags = matrix(2)

clustering.kmeans.matrix.find_clusters(points,clusters,tags)
        
print(tags)
