base_dir = string.get_path(arg[0])

m = matrix(1000,500):uniformf(0,1,random(1234))

--------------------------------------------------------------------------

aU,aS,aVT = stats.pca(m)

aR = stats.mean_centered(m, "col_major")
amRot = aR * aU

--------------------------------------------------------------------------

bT,bP,bR,bV,bS = stats.iterative_pca{ X = m, K = 10, }
assert(matrix.fromFilename(base_dir.."pca-T.mat.gz"):equals(bT,1e-01))
assert(matrix.fromFilename(base_dir.."pca-P.mat.gz"):equals(bP,1e-01))
assert(matrix.fromFilename(base_dir.."pca-R.mat.gz"):equals(bR,1e-01))

--------------------------------------------------------------------------

bS:adjust_range(0,1):equals(aS:slice({1},{bS:size()}):adjust_range(0,1))
