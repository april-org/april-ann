--
local funcs = get_table_from_dotted_string("clustering.kmeans.matrix",true)

local BSIZE=1024

-------------------
-- FIND CLUSTERS --
-------------------

-- fill vector T with the index of centroid associated to each sample
--
--@param X samples, NxD matrix (N rows D columns)
--@param C centroids, KxD matrix
--@param T vector de tags de talla N, si este punter val 0 no es fa res
function funcs.find_clusters(X,C,T,verbose)
  assert(X and isa(X,matrix), "A matrix needed as 1st argument")
  assert(C and isa(C,matrix), "A matrix needed as 2nd argument")
  local Xdim = X:dim()
  local Cdim = C:dim()
  assert(#Xdim == 2, "Data matrix must be bi-dimensional")
  assert(#Cdim == 2, "Centroids matrix must be bi-dimensional")
  assert(X:get_major_order() == C:get_major_order(),
	 "Given matrices with different major order")
  local N = Xdim[1] -- number of samples
  local D = Xdim[2] -- number of features (dimensions)
  local K = Cdim[1] -- number of clusters
  april_assert(Cdim[2] == D,
	       "Different columns found between data and centroids: %d ~= %d\n",
	       D, Cdim[2])
  local T = T or matrixInt32(N,1)
  april_assert(#T:dim() == 2 and T:dim(1) == N and T:dim(2) == 1 and isa(T,matrixInt32),
	       "The tags matrix must be bi-dimensional matrixInt32 and with size %dx1\n",
	       N)
  --
  local auxXblock,mins
  local Mnew       = matrix[X:get_major_order()]
  local M2Y        = Mnew(BSIZE,K)
  local CScores    = Mnew(K):zeros()
  local CSqScores  = Mnew(K):zeros()
  local Ccont      = matrixInt32(K):zeros()
  local score      = 0.0
  -- compute Ysquare
  local Ysq = C:clone():pow(2):sum(2) -- vector of size K
  -- traverse in BSIZE blocks
  for b=1,N,BSIZE do
    local block_size = math.min(BSIZE,N-b+1)
    if block_size ~= BSIZE then mins,auxXblock = nil,nil end
    local Xblock = X:slice({b,1},{block_size,D})
    local M2Yblock = M2Y:slice({1,1},{block_size,K})
    -- STEP 1: multiply Xblock by C to compute clusters distances
    M2Yblock:gemm{ A=Xblock, B=C, trans_A=false, trans_B=true,
		   alpha=-2.0, beta=0.0 }
    -- STEP 2: sums Y**2
    local M2Yrow
    for i=1,block_size do
      M2Yrow = M2Yblock:select(1,i,M2Yrow)
      M2Yrow:axpy(1.0,Ysq)
    end -- for i=1,block_size do
    -- STEP 3: compute closest centroid
    mins,argmins = M2Yblock:min(2,mins,T:slice({b,1},{block_size,1}))
    -- STEP 4: accumulate score
    auxXblock = auxXblock or Xblock:clone()
    local scores = auxXblock:copy(Xblock):pow(2):sum(2):axpy(1.0,mins)
    for i=1,K do
      local v   = scores:get(i,1)
      local pos = argmins:get(i,1)
      local aux = CScores:get(pos)
      CScores:set(pos, aux + v)
      aux = CSqScores:get(pos)
      CSqScores:set(pos, aux + v*v)
      Ccont:set(pos, Ccont:get(pos) + 1)
    end -- for i=1,K do
    -- sum all scores
    score = score + mins:sum()
  end -- for b=1,N,BSIZE
  for k=1,K do
    local sc,c = CScores:get(k),Ccont:get(k)
    local kdist = sc / c
    score = score + sc
    if verbose then
      printf("# Cluster %d, %d/%d (%0.3f%%), samples, dt: %0.9f\n",
	     k, c ,N, c*100.0/N, kdist)
    end
  end
  score = score/N
  return score,T
end

----------------------------
-- __call BASIC ALGORITHM --
----------------------------

function funcs.basic(X,C,params)
  local params = get_table_fields(
    {
      threshold = { mandatory=false, type_match="number", default=1e-5 },
      max_iter = { mandatory=false, type_match="number", default=100 },
      verbose = { mandatory=false },
    }, params)
  assert(X and isa(X,matrix), "A matrix needed as 1st argument")
  assert(C and isa(C,matrix), "A matrix needed as 2nd argument")
  local Xdim = X:dim()
  local Cdim = C:dim()
  assert(#Xdim == 2, "Data matrix must be bi-dimensional")
  assert(#Cdim == 2, "Centroids matrix must be bi-dimensional")
  assert(X:get_major_order() == C:get_major_order(),
	 "Given matrices with different major order")
  local N = Xdim[1] -- number of samples
  local D = Xdim[2] -- number of features (dimensions)
  local K = Cdim[1] -- number of clusters
  april_assert(Cdim[2] == D,
	       "Different columns found between data and centroids: %d ~= %d\n",
	       D, Cdim[2])
  local max_iter    = params.max_iter
  local threshold   = params.threshold
  local Mnew        = matrix[X:get_major_order()]
  local M2Y         = Mnew(BSIZE,K)
  local Csum        = Mnew(K,D)
  local Csq         = Mnew(K,D)
  local Ccont       = Mnew(K)
  local iter        = 0
  local score
  local discrepancy
  repeat
    local Csum_row,Csq_row,X_row,mins,argmins,auxXblock
    -- compute Ysquare
    local Ysq = C:clone():pow(2):sum(2) -- vector of size K
    -- counters set to zero
    Csum:zeros()
    Csq:zeros()
    Ccont:zeros()
    score       = 0.0
    discrepancy = 0.0
    -- traverse in blocks of BSIZE
    for b=1,N,BSIZE do
      local block_size = math.min(BSIZE,N-b+1)
      if block_size ~= BSIZE then mins,argmins,auxXblock = nil,nil,nil end
      local Xblock = X:slice({b,1},{block_size,D})
      local M2Yblock = M2Y:slice({1,1},{block_size,K})
      -- STEP 1: multiply Xblock by C to compute clusters distances
      M2Yblock:gemm{ A=Xblock, B=C, trans_A=false, trans_B=true,
		     alpha=-2.0, beta=0.0 }
      -- STEP 2: sums Y**2
      local M2Yrow
      for i=1,block_size do
	M2Yrow = M2Yblock:select(1,i,M2Yrow)
	M2Yrow:axpy(1.0,Ysq)
      end -- for i=1,block_size do
      -- STEP 3: compute closest centroid
      mins,argmins = M2Yblock:min(2,mins,argmins)
      -- STEP 4: accumulate score
      auxXblock = auxXblock or matrix.as(Xblock)
      auxXblock = auxXblock:copy(Xblock):pow(2)
      local scores = auxXblock:pow(2):sum(2):axpy(1.0,mins)
      score = score + mins:sum()
      
      for i=1,block_size do
	local cpos = argmins:get(i,1)
	Csum_row   = Csum:select(1,cpos,Csum_row)
	Csq_row    = Csq:select(1,cpos,Csq_row)
	X_row      = Xblock:select(1,i)
	X_row2     = auxXblock:select(1,i)
	Csum_row:axpy(1.0, X_row)
	Csq_row:axpy(1.0, X_row2)
	Ccont:set(cpos, Ccont:get(cpos) + 1)
      end
    end
    -- STEP 5: compute discrepancy with previous iteration
    for i=1,K do
      local count = Ccont:get(i)
      if count > 0 then
	local rcount = 1/count
	local csum_i,c_i = Csum(i,':'),C(i,':')
	csum_i:scal(rcount)
	Csq(i,':'):scal(rcount)
	local diff = (csum_i - c_i):abs():sum()
	discrepancy = discrepancy + diff
	c_i:copy(csum_i)
      end
    end
    --
    iter = iter + 1
    if params.verbose then
      printf("# Iteration %d. Centroids Discrepancy: %g\n",
	     iter,discrepancy)
    end
    io.stdout:flush()
  until iter == params.max_iter or discrepancy < params.threshold
  -- STEP 6: compute distortion
  if params.verbose then
    for i=1,K do
      printf("# Cluster %d: %d/%d samples ( %.3f%% )\n",
	     i,Ccont:get(i),N,Ccont:get(i)*100.0/N)
    end
  end
  local norm2 = X:norm2()
  score = (score + norm2*norm2) / N
  return score,C
end

-----------------------------
-- __call REFINE ALGORITHM --
-----------------------------

--[[
refine initialization for kmeans, see paper

@inproceedings{bradley1998refining,
  title={Refining initial points for k-means clustering},
  author={Bradley, P.S. and Fayyad, U.M.},
  booktitle={Proceedings of the Fifteenth International Conference on Machine Learning},
  volume={66},
  year={1998},
  organization={San Francisco, CA, USA}
}

matrix C does NOT contain centroids, it is used to return the
resulting centroids

returns best distortion
--]]

function funcs.refine(X,C,params)
  local params = get_table_fields(
    {
      subsamples = { mandatory=false, type_match="number", default=10 },
      percentage = { mandatory=false, type_match="number", default=0.01 },
      random = { mandatory=true, isa_match=random },
      threshold = { mandatory=false, type_match="number", default=1e-5 },
      max_iter = { mandatory=false, type_match="number", default=100 },
      verbose = { mandatory=false },
    }, params)
  assert(X and isa(X,matrix), "A matrix needed as 1st argument")
  assert(C and isa(C,matrix), "A matrix needed as 2nd argument")
  local Xdim = X:dim()
  local Cdim = C:dim()
  assert(#Xdim == 2, "Data matrix must be bi-dimensional")
  assert(#Cdim == 2, "Centroids matrix must be bi-dimensional")
  assert(X:get_major_order() == C:get_major_order(),
	 "Given matrices with different major order")
  local N = Xdim[1] -- number of samples
  local D = Xdim[2] -- number of features (dimensions)
  local K = Cdim[1] -- number of clusters
  local J = params.subsamples
  local rnd = params.random
  april_assert(Cdim[2] == D,
	       "Different columns found between data and centroids: %d ~= %d\n",
	       D, Cdim[2])
  local num_samples = math.max(K, math.round(N * params.percentage))
  local Mnew = matrix[X:get_major_order()]
  local S = Mnew(num_samples, D)
  local CM = Mnew(J, K, D)
  local FM = matrix.as(CM)
  local function get_row_idx() return rnd:randInt(1,N) end
  -- auxiliary variables which store temporal matrices
  local Xrow,Crow,Srow,CMiter,FMiter
  -- cluster initialization
  for c=1,K do
    local row = get_row_idx()
    Xrow = X:select(1,row,Xrow)
    Crow = C:select(1,c,Crow)
    Crow:copy(Xrow)
  end
  -- compute J kmeans subsamples
  for j=1,J do
    -- subsampling
    for s=1,num_samples do
      local row = get_row_idx()
      -- copy random row to matrix S
      Xrow = X:select(1,row,Xrow)
      Srow = S:select(1,s,Srow)
      Srow:copy(Xrow)
    end
    -- copy centroids from C to CM
    CMiter = CM:select(1,j,CMiter)
    CMiter:copy(C)
    if params.verbose then printf("# REFINE: %d J %d\n", num_samples, j) end
    io.stdout:flush()
    funcs.basic(S,CMiter,{ max_iter=params.max_iter,
			   threshold = params.threshold,
			   verbose = params.verbose })
  end
  -- here we have J different sets of K centroids in matrix CM
  FM:copy(CM)
  -- look for the best set
  FMiter = FM:select(1,1,FMiter)
  local best_score = funcs.basic(X,FMiter,{verbose = params.verbose})
  local best_J = 1
  for j=2,J do
    FMiter = FM:select(1,j,FMiter)
    local score = funcs.basic(X,FMiter,{verbose = params.verbose})
    if score < best_score then best_score,best_J = score,j end
  end
  -- copy the best centroids to C
  C:copy(FM:select(1,best_J,FMiter))
  return best_score,C
end

-----------------------
-- __call METAMETHOD --
-----------------------

local function metatable_call(self,params)
  local params = get_table_fields(
    {
      data = { mandatory=true, isa_match=matrix, default=nil },
      K = { mandatory=false, type_match="number", default=nil },
      distance = { mandatory=false, type_match="string", default="euclidean" },
      subsamples = { mandatory=false, type_match="number" },
      percentage = { mandatory=false, type_match="number" },
      random = { mandatory=false, isa_match=random, default=nil },
      centroids = { mandatory=false, isa_match=matrix, default=nil },
      threshold = { mandatory=false, type_match="number" },
      max_iter = { mandatory=false, type_match="number" },
      verbose = { mandatory=false },
    }, params)
  -- sanity checks
  april_assert(params.distance == "euclidean",
	       "Distance %s not supported\n", params.distance)
  assert(params.K or params.centroids,
	 "Parameter K or centroids must be defined")
  if params.K and params.centroids then
    assert(params.centroids:dim(1) == params.K,
	   "Centroids matrix dim(1) is different of given K parameter")
  end
  --
  local centroids = params.centroids
  local data = params.data
  local distortion
  if not centroids then
    assert(random, "Field random is mandatory when not given centroids")
    centroids = matrix[data:get_major_order()](params.K,data:dim(2))
    distortion = funcs.refine(data, centroids, {
				max_iter   = params.max_iter,
				random     = params.random,
				threshold  = params.threshold,
				percentage = params.percentage,
				subsamples = params.subsamples,
				verbose    = params.verbose })
  end
  distortion = funcs.basic(data, centroids, {
			     max_iter  = params.max_iter,
			     threshold = params.threshold,
			     verbose   = params.verbose })
  return distortion,centroids
end

setmetatable(funcs, { __call = metatable_call })
	     
return funcs
