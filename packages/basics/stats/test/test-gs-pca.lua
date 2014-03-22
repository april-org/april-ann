base_dir = string.get_path(arg[0])

function compute_cov(X,dim)
  local other = 3 - dim
  local major_order = X:get_major_order()
  return matrix[major_order](X:dim(dim),X:dim(dim)):gemm{ A=X, B=X,
							  trans_A=true,
							  alpha=1/X:dim(other),
							  beta=0 }
end

--------------------------------------------------------------------------

ok,m = pcall(matrix.fromTabFilename, "/tmp/sample.txt.gz")

if not ok then
  ok=os.execute("curl -0 http://cafre.dsic.upv.es:8080/~pako/STUFF/sample.txt.gz > /tmp/sample.txt.gz")
  if not ok then
    print("WARNING: impossible to run test\n")
    os.exit(0)
  end
  if not io.open("/tmp/sample.txt.gz") then
    print("Ignoring test, impossible to connect with cafre.dsic.upv.es")
    os.exit(0)
  end
  m = matrix.fromTabFilename("/tmp/sample.txt.gz"):transpose()
else
  m = m:transpose()
end


--------------------------------------------------------------------------

aR = stats.mean_centered_by_pattern(m:clone("col_major"))

aU,aS,aVT = stats.pca(aR)

-- check regeneration of original covariance matrix
cov = compute_cov(aR, 2)
assert(cov:equals( aU * aS:to_dense("col_major") * aVT, 1e-04 ))

-- ROTATION
amRot = aR * aU
-- check covariance of rotated data
cov = compute_cov(amRot, 2)
-- adjusting the data to be between 0 and 1
for sw in cov:sliding_window():iterate() do sw:adjust_range(0,1) end
-- the adjusted covariance must be an identity matrix
-- assert(cov:equals( cov:clone():zeros():diag(1), 0.1 ))

-- U matrix orthogonality check
aUmul = aU:clone():gemm{ A=aU, B=aU, trans_B=true, alpha=1.0, beta=0.0, }
assert(aUmul:equals( aUmul:clone():zeros():diag(1), 0.001 ))
-- V matrix orthogonality check
aVTmul = aVT:clone():gemm{ A=aVT, B=aVT, trans_B=true, alpha=1.0, beta=0.0, }
assert(aVTmul:equals( aVTmul:clone():zeros():diag(1), 0.001 ))

-- check U matrix with octave computation
refU = matrix.fromTabFilename(base_dir.."data/U.gz", "col_major"):abs()
assert(refU:equals(aU:clone():abs(), 0.001))

-- check V matrix with octave computation
refV = matrix.fromTabFilename(base_dir.."data/V.gz", "col_major"):transpose():abs()
assert(refV:equals(aVT:clone():abs(), 0.001))

-- check S matrix with octave computation
refS = matrix.fromFilename(base_dir.."data/S.gz", "col_major")
assert(refS:diagonalize():equals(aS:to_dense("col_major"), 0.001))

--------------------------------------------------------------------------
--------------------------------------------------------------------------
--------------------------------------------------------------------------

aR = stats.mean_centered_by_pattern(m:clone("col_major"))
bT,bP,bR,bV,bS = stats.iterative_pca{ X = aR, K = 144, }

-- check regeneration of original matrix
assert(aR:equals( bT * bP:transpose() + bR, 1e-04 ))

-- check covariance of rotated data
cov = compute_cov(bT, 2)
-- adjusting the data to be between 0 and 1
for sw in cov:sliding_window():iterate() do sw:adjust_range(0,1) end
-- the adjusted covariance must be an identity matrix
-- assert(cov:equals( cov:clone():zeros():diag(1), 1e-02 ))

-- U matrix orthogonality check
bPmul = bP:clone():gemm{ A=bP, B=bP, trans_B=true, alpha=1.0, beta=0.0, }
assert(bPmul:equals( bPmul:clone():zeros():diag(1), 1e-03 ))
