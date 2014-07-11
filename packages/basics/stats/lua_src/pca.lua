stats = get_table_from_dotted_string("stats", true)

local function normalize(m,norm)
  local norm = norm or m:norm2()
  m:scal(1/norm)
end

local function re_orhtonormalize(submatrix,colmatrix,aux)
  aux:gemv{ alpha=1.0, A=submatrix, X=colmatrix, beta=0.0, trans_A=true }
  colmatrix:gemv{ alpha=-1.0, A=submatrix, X=aux, beta=1.0, trans_A=false }
end

-------------------------------------------------------------------------------

-- WARNING IN PLACE OPERATION
function stats.mean_centered_by_pattern(X)
  local dim = X:dim()
  assert(#dim == 2, "Expected a bi-dimensional matrix")
  local M,N = table.unpack(dim)
  -- U is the mean over all rows
  local U,auxX = X:sum(2):rewrap(M):scal(1/N)
  -- X is centered subtracting by -U
  for i=1,X:dim(2) do auxX=X:select(2,i,auxX):axpy(-1, U) end
  return X,U
end

-------------------------------------------------------------------------------

-- NOT IN-PLACE
function stats.pca_whitening(X,U,S,epsilon)
  local epsilon = epsilon or 0.0
  local result = matrix[X:get_major_order()](X:dim(1), S:dim(1))
  result:gemm{ A=X, B=U, trans_B=false, beta=0, alpha=1}
  for v,i in S:iterate() do
    result:select(2,i):scal( 1/math.sqrt(v + epsilon) )
  end
  return result
end

-- WARNING IN PLACE OPERATION
function stats.zca_whitening(X,U,S,epsilon)
  local aux = stats.pca_whitening(X,U,S,epsilon)
  X:gemm{ A=aux, B=U, trans_B=true, beta=0, alpha=1 }
  return X
end

-- compute PCA mass
function stats.pca_mass(S,mass)
  local acc = 0
  local acc_mat = matrix.col_major(S:size())
  for v,i in S:iterate() do
    acc=acc + v
    acc_mat:set(i, acc)
  end
  acc_mat:scal(1.0/acc)
  return acc_mat
end

-- show PCA threshold
function stats.pca_threshold(S,mass)
  local mass = mass or 0.99
  local acc = 0
  local sum = S:sum()
  local acc_th,th,vth = 0,1
  for v,i in S:iterate() do
    acc=acc + v
    if acc/sum < mass then vth,acc_th,th=v,acc,i end
  end
  assert(acc_th > 0, "The probability mass needs to be larger")
  return th,vth,acc_th/sum
end

-------------------------------------------------------------------------------

-- PCA algorithm based on covariance matrix and SVD decomposition the matrix Xc
-- must be zero mean centerd for each pattern. Patterns are ordered by rows.
function stats.pca(Xc)
  local dim    = Xc:dim()
  assert(#dim == 2, "Expected a bi-dimensional matrix")
  local aux = Xc:sum(2):scal(1/Xc:dim(2)):rewrap(Xc:dim(1))
  local M,N    = table.unpack(dim)
  local sigma  = matrix.col_major(N,N)
  sigma:gemm{ A=Xc, B=Xc,
	      trans_A=true,
	      trans_B=false,
	      alpha=1/M,
	      beta=0, }
  local U,S,VT = sigma:svd()
  return U,S,VT
end

-------------------------------------------------------------------------------

april_set_doc("stats.iterative_pca",
	      {
		class = "function",
		summary = "Computes PCA using GS-PCA (iterative PCA algorithm)",
		params = {
		  X = "A matrix of MxN, M are numPatterns and N patternSize",
		  K = "The number of components that you want, K <= N",
		  max_iter = "Maximum number of iterations [optional], 10000 by default",
		  epsilon  = "A number with the convergence criterion [optional], by default 1e-07",
		},
		outputs = {
		  "The T=V*S=X*U scores matrix, size MxK",
		  "The P loads matrix, or U right eigenvectors matrix, size NxK",
		  "The R residuals matrix, size MxN",
		  "The V left eigenvectors matrix, size MxN",
		  "The S singular values vector, size K",
		},
	      })
-- EXTRACTED FROM:
-- Parallel GPU Implementation of Iterative PCA Algorithms, M. Andrecut
--
-- http://arxiv.org/pdf/0811.1081.pdf
-- 

-- GS-PCA algorithm
--
-- input: X is a MxN matrix, M number of patterns, N pattern size
-- input: K is the number of components, K <= N
-- input: max_iter is the maximum number of iterations
-- input: epsilon is the convergence criterion
-- output: T is a MxK scores matrix
-- output: P is a NxK loads matrix
-- output: R is a MxN residuals matrix
--
-- PCA model: X = TLP’ + R
function stats.iterative_pca(params)
  local params = get_table_fields(
    {
      X = { isa_match=matrix, mandatory=true },
      K = { type_match="number", mandatory=true },
      max_iter = { type_match="number", mandatory=true, default=10000 },
      epsilon  = { type_match="number", mandatory=true, default=1e-07 },
    },
    params)
  local X,K,max_iter,epsilon = params.X,params.K,params.max_iter,params.epsilon
  local M,N = table.unpack(X:dim())
  assert(K<=N, "K <= N failed")
  assert(#X:dim() == 2, "Needs a bi-dimensional matrix")
  if M < N then
    print("# Warning, M < N, probably matrix need to be transposed")
  end
  local major_order = X:get_major_order()
  local T = matrix[major_order](M,K):zeros() -- left eigenvectors
  local P = matrix[major_order](N,K):zeros() -- right eigenvectors
  local L = matrix[major_order](K):zeros()   -- eigenvalues
  local R = X:clone()
  local U = R:sum(2):scal(1/R:dim(2)):rewrap(R:dim(1))
  assert( math.abs(U:sum() / U:size()) < 1e-03,
	  "A zero mean (at each pattern) data matrix is needed")
  -- GS-PCA
  local Tcol, Rcol, Pcol, Uslice, Pslice, Tslice, Lk
  for k=1,K do
    Tcol = T:select(2,k,Tcol)
    Rcol = R:select(2,k,Rcol)
    Pcol = P:select(2,k,Pcol)
    --
    if k > 1 then
      Uslice = U:slice({1},{k-1})
      Pslice = P:slice({1,1},{N,k-1})
      Tslice = T:slice({1,1},{M,k-1})
    end
    --
    Tcol:copy(Rcol)
    local a = 0.0
    for it=1,max_iter do
      Pcol:gemv{ alpha=1.0, A=R, X=Tcol, beta=0.0, trans_A=true }
      if k>1 then re_orhtonormalize(Pslice, Pcol, Uslice) end
      normalize(Pcol)
      Tcol:gemv{ alpha=1.0, A=R, X=Pcol, beta=0.0, trans_A=false }
      if k>1 then re_orhtonormalize(Tslice, Tcol, Uslice) end
      Lk = Tcol:norm2()
      normalize(Tcol, Lk)
      if math.abs(a - Lk) < epsilon*Lk then break end
      a = Lk
    end
    L:set(k, Lk)
    R:ger{ alpha=-Lk, X=Tcol, Y=Pcol }
  end
  local V = T:clone()
  for k=1,K do T:select(2,k):scal( L:get(k) ) end
  return T,P,R,V,L
end
