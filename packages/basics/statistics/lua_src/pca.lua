stats = get_table_from_dotted_string("stats", true)

local function normalize(m,norm)
  local norm = norm or m:norm2()
  m:scal(1/norm)
end

local function re_orhtonormalize(submatrix,colmatrix,aux)
  aux:gemv{ alpha=1.0, A=submatrix, X=colmatrix, beta=0.0, trans_A=true }
  colmatrix:gemv{ alpha=-1.0, A=submatrix, X=aux, beta=1.0, trans_A=false }
end

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
		  "The T scores matrix, size MxK",
		  "The P loads matrix, size NxK",
		  "The R residuals matrix, size MxN",
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
  local R = X:clone()                        -- residual
  -- U is the sum over all columns
  local U,auxR = R:sum(2):rewrap(M)
  -- R is centered subtracting by -1/N*U
  for i=1,R:dim(2) do auxR=R:select(2,i,auxR):axpy(-1/N, U) end
  -- GS-PCA
  local Tcol, Rcol, Pcol, Uslice, Pslice, Tslice, Lk
  for k=1,K do
    Tcol = T:select(2,k)
    Rcol = R:select(2,k)
    Pcol = P:select(2,k)
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
  for k=1,K do T:select(2,k):scal( L:get(k) ) end
  return T,P,R
end
