local assert = assert
local ipairs = ipairs
local math = math
local pairs = pairs
local string = string
local table = table
local type = type
--
local april_assert = april_assert
local get_table_fields = get_table_fields
local iterator = iterator
local mop = matrix.op
local md = matrix.dict

local MAX_UPDATES_WITHOUT_PRUNE = ann.optimizer.MAX_UPDATES_WITHOUT_PRUNE

--------------------------------------------
--------- SIMPLEX ARMIJO DOWNHILL ----------
--------------------------------------------

-- "A Simplex Armijo Downhill Algorithm for Optimizing Statistical"
-- Machine Translation Decoding Parameters"
-- Bing Zhao, Shengyuan chen
-- IBM T.J. Watson Research

local simplex, simplex_methods = class("ann.optimizer.simplex", ann.optimizer)
ann.optimizer.simplex = simplex -- global environment

function simplex:constructor(g_options, l_options, count,
                             lambda0, lambdaR, lambdaE, lambdaC,
                             lambdaP, candidates)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
			      {"beta", "Random perturbation of simplex movements (0.025)"},
			      {"rand", "Random numbers generator (random())"},
                              {"max_iter", "Maximum number of iterations (100)"},
                              {"epsilon", "Simplex size stop (1e-5)"},
                              {"tol", "Limit of rel. dif. between best and worst (1e-5)"},
                              {"alpha", "Reflexion coef (1)"},
                              {"gamma", "Expansion coef (2)"},
                              {"rho", "Shrink coef (0.5)"},
                              {"weight_decay", "Weight L2 regularization (0.0)"},
                              {"L1_norm", "Weight L1 regularization (0.0)"},
                              {"max_norm_penalty", "Weight max norm upper bound (0)"},
                              {"verbose", "true or false (false)"},
			    },
			    g_options,
			    l_options,
			    count)
  self.lambda0 = lambda0
  self.lambdaR = lambdaR
  self.lambdaE = lambdaE
  self.lambdaC = lambdaC
  self.lambdaP = lambdaP
  self.candidates = candidates
  if not g_options then
    -- default values
    self:set_option("beta", 0.05)
    self:set_option("rand", random())
    self:set_option("max_iter", 100)
    self:set_option("epsilon", 1e-05)
    self:set_option("tol", 1e-05)
    self:set_option("alpha", 1)
    self:set_option("gamma", 2)
    self:set_option("rho", 0.5)
    self:set_option("weight_decay", 0.0)
    self:set_option("L1_norm", 0.0)
    self:set_option("max_norm_penalty", 0.0)
    self:set_option("verbose", false)
  end
end

function simplex_methods:execute(eval, weights)
  local table = table
  local assert = assert
  -- DO EVAL
  local counter = 0
  local do_eval = function(x)
    local loss = eval(x, counter)
    counter = counter + 1
    local reg = 0
    for wname,w in pairs(x) do
      local l1 = self:get_option_of(wname, "L1_norm")
      local l2 = self:get_option_of(wname, "weight_decay")
      if l1 > 0.0 then reg = reg + l1*mop.abs(w):sum() end
      if l2 > 0.0 then reg = reg + 0.5*l2*w:dot(w) end
    end
    return loss + reg
  end
  -- APPLY PENALTIES
  local apply_penalties = function(x)
    for wname,w in pairs(x) do
      local mnp = self:get_option_of(wname, "max_norm_penalty")
      -- constraints
      if mnp > 0.0 then ann.optimizer.utils.max_norm_penalty(w, mnp) end
    end
    if self:get_count() % MAX_UPDATES_WITHOUT_PRUNE == 0 then
      md.prune_subnormal_and_check_normal( x )
    end
  end
  local beta     = self:get_option("beta")
  local rand     = self:get_option("rand")
  local epsilon  = self:get_option("epsilon")
  local tol      = self:get_option("tol")
  local alpha    = self:get_option("alpha")
  local gamma    = self:get_option("gamma")
  local rho      = self:get_option("rho")
  local max_iter = self:get_option("max_iter")
  local verbose  = self:get_option("verbose")
  --
  local sz = md.size(weights)
  local lambda0 = self.lambda0 or md.clone_only_dims(weights)
  local lambdaR = self.lambdaR or md.clone_only_dims(weights)
  local lambdaE = self.lambdaE or md.clone_only_dims(weights)
  local lambdaC = self.lambdaC or md.clone_only_dims(weights)
  local lambdaP = self.lambdaP or md.clone_only_dims(weights)
  local candidates = self.candidates
  --
  local function sort_candidates(candidates)
    table.sort(candidates, function(a,b) return a[2] < b[2] end)
  end
  --
  if not candidates then
    candidates = {}
    for i=1,sz+1 do candidates[i] = { md.clone(weights), math.huge } end
    self.candidates = candidates
    -- 1 is the base point of simplex (initial simplex)
    -- random perturbations for rest of candidates (initial simplex)
    for i=2,#candidates do
      for _,m in pairs(candidates[i][1]) do
        m:axpy(1.0, matrix(table.unpack(m:dim())):uniformf(-beta, beta, rand))
      end
    end
  end
  -- evaluate candidates with new samples
  for i=1,#candidates do candidates[i][2] = do_eval(candidates[i][1]) end
  sort_candidates(candidates)
  --
  local function point(i) return candidates[i][1] end
  local function loss(i) return candidates[i][2] end
  local function compose(D, A, alpha, B, C)
    for d,a,b,c in iterator.zip(iterator(D), iterator(A),
                                iterator(B), iterator(C)) do
      d[{}] = a + alpha * (b - c)
    end
    return D
  end
  --
  self:count_one()
  local iter = 0
  repeat
    iter = iter + 1
    local simplex_module = 0
    for i=1,#candidates do
      simplex_module = simplex_module + md.norm2(point(i))
    end

    if simplex_module < epsilon then break end
    --
    -- indicate if a shrink has been produced
    local shrinked = false
    -- tolerance control, difference between best an worst candidate
    local rel_tol = 2.0 * math.abs(loss(#candidates) - loss(1))/math.abs(loss(#candidates) + loss(1))
    if rel_tol < tol then break end
    -- compute centroid of all points ignoring the worst
    md.zeros(lambda0)
    for i=1,#candidates-1 do md.axpy(lambda0, 1.0, point(i)) end
    md.scal(lambda0, 1/sz)
    -- compute reflexion of centroid over worst point
    compose(lambdaR, lambda0, alpha, lambda0, point(#candidates))
    
    local S1 = loss(1)
    local SR = do_eval(lambdaR)
    local SM = loss(#candidates - 1)
    
    if S1 <= SR and SR <= SM then
      -- if reflected point is between best and second worst, we take it
      md.copy(lambdaP, lambdaR)
    elseif SR < S1 then
      -- if reflected point improves the best, then try expansion
      compose(lambdaE, lambda0, gamma, lambda0, point(#candidates))
      local SE = do_eval(lambdaE)
      -- if expansion improves, take it
      if SE < SR then md.copy(lambdaP, lambdaE)
      else -- otherwise, take reflection
        md.copy(lambdaP, lambdaR)
      end
    elseif SR > SM then
      -- the reflected point not improves second worst, then try shrink
      compose(lambdaC, point(#candidates), rho, lambda0, point(#candidates))
      local SC = do_eval(lambdaC)
      -- if shrink improves reflected, take it
      if SC < SR then
        md.copy(lambdaP, lambdaC)
      else
        -- impossible to improve :S, shrink whole simplex around the best
        for i=2,#candidates do
          md.axpy(candidates[i][1], 1.0, candidates[1][1])
          md.scal(candidates[i][1], 0.5)
          candidates[i][2] = do_eval(candidates[i][1])
        end
        shrinked = true
      end
    end
    
    if not shrinked then
      local prev_loss = loss(#candidates)
      -- replace worst point
      md.copy(point(#candidates), lambdaP)
      candidates[#candidates][2] = do_eval(candidates[#candidates][1])
      if prev_loss == loss(#candidates) then
        -- warning: alpha=1 causes infinite loops of not improving
        alpha = alpha * 0.9
      end
    end

    sort_candidates(candidates)
    
    if verbose then
      printf("# %6d %6d %.6f %.6f %.6f\n",
             self:get_count(), iter,
             loss(1), loss(#candidates),
             simplex_module)
    end
    
  until iter >= max_iter
  --
  self.lambda0 = lambda0
  self.lambdaR = lambdaR
  self.lambdaE = lambdaE
  self.lambdaC = lambdaC
  self.lambdaP = lambdaP
  --
  md.copy(weights, point(1))
  apply_penalties(weights)
  --
  return eval(weights, iter)
end

function simplex_methods:clone()
  local obj = ann.optimizer.simplex()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  obj.candidates        = util.clone( self.candidates )
  obj.lambda0           = util.clone( self.lambda0 )
  obj.lambdaR           = util.clone( self.lambdaR )
  obj.lambdaE           = util.clone( self.lambdaE )
  obj.lambdaC           = util.clone( self.lambdaC )
  obj.lambdaP           = util.clone( self.lambdaP )
  return obj
end

function simplex_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.simplex(",
		  table.tostring(self.global_options),
		  ",",
		  table.tostring(self.layerwise_options),
		  ",",
		  tostring(self.count),
		  ",",
		  util.to_lua_string(self.lambda0, format),
		  ",",
		  util.to_lua_string(self.lambdaR, format),
		  ",",
		  util.to_lua_string(self.lambdaE, format),
		  ",",
		  util.to_lua_string(self.lambdaC, format),
		  ",",
		  util.to_lua_string(self.lambdaP, format),
		  ",",
		  util.to_lua_string(self.candidates, format),
		  ")" }
  return table.concat(str_t, "")
end
