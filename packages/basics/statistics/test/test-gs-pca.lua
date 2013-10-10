m = matrix(1000,500):uniformf(0,1,random(1234))
T,P,R = stats.iterative_pca{ X = m, K = 10, }
