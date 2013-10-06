stats = get_table_from_dotted_string("stats", true)

-- EXTRACTED FROM: http://arxiv.org/pdf/0811.1081.pdf
-- 

-- input: X is a MxN matrix (M is number of patterns, N is pattern size)
-- input: K is the number of components
-- output: T, KxM scores matrix
-- output: P, KxN loads matrix
-- output: R, MxN residuals matrix
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
  local N,M = table.unpack(X:dim())
  assert(#X:dim() == 2, "Needs a bi-dimensional matrix")
  assert(X:get_major_order() == "col_major", "Needs a col-major matrix")
  local T = matrix.col_major(K,M):zeros()
  local P = matrix.col_major(K,N):zeros()
  local R = X:clone()
  -- the eigen values
  local L = matrix.col_major(K):zeros()
  -- mean centered data
  local U,auxR = R:sum(1)
  for i=1,R:dim(1) do auxR=R:select(1,i,auxR):axpy(-1/N, U) end
  -- GS-PCA
  local Trow, Rrow, Prow, Uslice, Pslice, Tslice, Lk
  for k=1,K do
    Trow = T:slice({k,1},{1,M})
    Rrow = R:slice({k,1},{1,M})
    Prow = P:slice({k,1},{1,N})
    Uslice = U:slice({1,1},{1,k})
    Pslice = P:slice({1,1},{k,N})
    Tslice = T:slice({1,1},{k,M})
    --
    Trow:copy(Rrow)
    local a = 0.0
    for it=1,max_iter do
      print(k, it)
      print(Prow)
      print("+++++++++++++++++++++++++++++*")
      Prow:gemv{ alpha=1.0, A=R, trans_A=false, X=Trow, beta=0.0 }
      if k>1 then
	Uslice:gemv{ alpha=1.0, A=Pslice, X=Prow, beta=0.0, trans_A=false }
	Prow:gemv{ alpha=-1.0, A=Pslice, X=Uslice, beta=1.0, trans_A=true }
      end
      print("NORM2",Prow:norm2())
      if Prow:norm2() > 100 then
	print(P)
	print(U)
      end
      Prow:scal( 1.0/Prow:norm2() )
      Trow:gemv{ alpha=1.0, A=R, X=Prow, beta=0.0, trans_A=true }
      if k>1 then
	Uslice:gemv{ alpha=1.0, A=Tslice, X=Trow, beta=0.0, trans_A=false }
	Trow:gemv{ alpha=-1.0, A=Tslice, X=Uslice, alpha=1.0, trans_A=true }
      end
      Lk = Trow:norm2()
      L:set(k, Lk)
      Trow:scal(1/Lk)
      if math.abs(a - Lk) < epsilon*Lk then break end
      a = Lk
    end
    print("--------------------------")
    print(Prow)
    print(Trow)
    print(R)
    R:ger{ alpha=-Lk, X=Prow, Y=Trow }
    print(R)
  end
  for k=1,K do T:slice({k,1},{1,M}):scal( L:get(k) ) end
  return T,P,R
end
