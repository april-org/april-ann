base_dir = string.get_path(arg[0])
m = matrix(1000,500):uniformf(0,1,random(1234))
T,P,R = stats.iterative_pca{ X = m, K = 10, }
assert(matrix.fromFilename(base_dir.."pca-T.mat.gz"):equals(T))
assert(matrix.fromFilename(base_dir.."pca-P.mat.gz"):equals(P))
assert(matrix.fromFilename(base_dir.."pca-R.mat.gz"):equals(R))
