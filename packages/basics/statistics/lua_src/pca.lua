stats = get_table_from_dotted_string("stats", true)

-- EXTRACTED FROM: http://arxiv.org/pdf/0811.1081.pdf
-- 

-- input: X is a MxN matrix
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
  assert(#X:dim() == 2, "Needs a bi-dimensional matrix")
  if M < N then
    print("# Warning, M < N, probably matrix need to be transposed")
  end
  local major_order = X:get_major_order()
  local T = matrix[major_order](matrix,M,K):zeros() -- left eigenvectors
  local P = matrix[major_order](matrix,N,K):zeros() -- right eigenvectors
  local L = matrix[major_order](matrix,K,1):zeros() -- eigenvalues
  local R = X:clone()                        -- residual
  -- U is the sum over all columns
  local U,auxR = R:sum(2)
  -- R is centered subtracting by -1/N*U
  for i=1,R:dim(2) do auxR=R:select(2,i,auxR):axpy(-1/N, U) end
  -- GS-PCA
  local Tcol, Rcol, Pcol, Uslice, Pslice, Tslice, Lk
  for k=1,K do
    Tcol = T:select(2,k)
    Rcol = R:select(2,k)
    Pcol = P:select(2,k)
    --
    Uslice = U:slice({1,1},{k,1})
    Pslice = P:slice({1,1},{N,k})
    Tslice = T:slice({1,1},{M,k})
    --
    Tcol:copy(Rcol)
    local a = 0.0
    for it=1,max_iter do
      Pcol:gemv{ alpha=1.0, A=R, trans_A=true, X=Tcol, beta=0.0 }
      if k>1 then
	Uslice:gemv{ alpha=1.0, A=Pslice, X=Pcol, beta=0.0, trans_A=true }
	print(k,it,"U0",Uslice:norm2())
	Pcol:gemv{ alpha=-1.0, A=Pslice, X=Uslice, beta=1.0, trans_A=false }
	print(k,it,"P",Pcol:norm2())
      end
      Pcol:scal( 1.0/Pcol:norm2() )
      print(k,it,Tcol:norm2())
      Tcol:gemv{ alpha=1.0, A=R, X=Pcol, beta=0.0, trans_A=false }
      print(k,it,Tcol:norm2())
      if k>1 then
	Uslice:gemv{ alpha=1.0, A=Tslice, X=Tcol, beta=0.0, trans_A=true }
	print(k,it,"U1",Uslice:norm2())
	Tcol:gemv{ alpha=-1.0, A=Tslice, X=Uslice, alpha=1.0, trans_A=false }
	print(k,it,Tcol:norm2())
      end
      print("--------------------------------------------------")
      Lk = Tcol:norm2()
      Tcol:scal(1/Lk)
      if math.abs(a - Lk) < epsilon*Lk then break end
      a = Lk
    end
    L:set(k, 1, Lk)
    R:ger{ alpha=-Lk, X=Tcol, Y=Pcol }
  end
  for k=1,K do T:select(2,k):scal( L:get(k) ) end
  return T,P,R
end
