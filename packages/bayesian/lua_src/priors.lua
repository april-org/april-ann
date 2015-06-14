get_table_from_dotted_string("bayesian", true)

------------------------------------------------------------------------------

local make_value = function(value)
  return { t="value", value=value }
end

local make_dist = function(dist, arg)
  return { t="dist", dist=dist, arg=arg }
end

local make_transform = function(func, arg)
  return { t="transform", func=func, arg=arg }
end

local normalize_shape = function(d, w)
  if w then
    if d:size() == 1 and w:dim(2) ~= 1 then
      return w:rewrap(w:size(),1)
    else
      april_assert(d:size() == w:dim(2),
                   "Incompatible matrix size, found %dx%d, expected %dx%d",
                   w:dim(1), w:dim(2), w:dim(1), d:size())
    end
  end
  return w
end

local sampling_funcs = {
  value = function(v) return function() return v.value end end,
  dist = function(v, outcomes)
    local arg = {}
    for i=1,#v.arg do
      arg[i] = april_assert(outcomes[v.arg[i]],
                            "Undefined value of '%s'", v.arg[i])
    end
    local d = stats.dist[v.dist](table.unpack(arg))
    v.obj = d
    return function(rng, w)
      local w = normalize_shape(d, w)
      return d:sample(rng, w)
    end
  end,
  transform = function(v, outcomes)
    local arg = {}
    for i=1,#v.arg do
      arg[i] = april_assert(outcomes[v.arg[i]],
                            "Undefined value of '%s'", v.arg[i])
      if isa(arg[i], matrix) then arg[i] = arg[i]:clone() end
    end
    local r = v.func(table.unpack(arg))
    return function() return r end
  end,
}

-----------------------------------------------------------------------------

local priors_class,priors_methods = class("bayesian.priors")
bayesian.priors = priors_class -- global environment

function priors_class:constructor(tree,order,outcomes)
  self.tree  = tree or {}
  self.order = order or {}
  self.outcomes = outcomes or {}
end

function priors_methods:compute_neg_log_prior(weights)
  error("NOT IMPLEMENTED")
  local logprob = 0.0
  for name,m in pairs(self.outcomes) do
    local d = self.tree[name].obj
    if d then
      local out = d:logpdf(m)
      logprob = logprob + out:sum()
    end
  end
  for name,w in pairs(weights) do
    local v = self.tree[name]
    if v and v.obj then
      local d = v.obj
      local w = normalize_shape(d, w)
      local out = d:logpdf(w)
      logprob = logprob + out:sum()
    end
  end
  return -logprob
end

function priors_methods:ctor_name() return "bayesian.priors" end
function priors_methods:ctor_params()
  return self.tree,self.order,self.outcomes
end

function priors_methods:value(name, value)
  april_assert(not self.tree[name], "Redifition of value '%s'", target)
  table.insert(self.order, name)
  self.tree[name] = make_value(value)
  return name
end

function priors_methods:follows(target, dist, ...)
  assert(type(dist) == "string", "Expected a string with the distribution name")
  april_assert(stats.dist[dist], "Unknown distribution %s\n", dist)
  april_assert(not self.tree[target], "Redifition of target '%s' dist", target)
  table.insert(self.order, target)
  self.tree[target] = make_dist(dist, { ... })
  return target
end

function priors_methods:transform(target, func, ...)
  april_assert(not self.tree[target], "Redifition of trans. '%s'", target)
  table.insert(self.order, target)
  self.tree[target] = make_transform(func, { ... })
  return target
end

function priors_methods:sample(rng, weights)
  local outcomes = {}
  for _,name in ipairs(self.order) do
    local v = assert( self.tree[name] )
    local s = sampling_funcs[v.t](v, outcomes)
    local w = weights(name)
    if not w then
      outcomes[name] = s(rng, w)
    end
    -- if w then w:copy(outcomes[name]) end
  end
  self.outcomes = outcomes
  return outcomes
end

function priors_methods:update_gradients_with_priors(weights, grads)
  for name,w in pairs(weights) do
    local wgrads = assert( grads(name) )
    local v = self.tree[name]
    if v then
      assert(v.t == "dist", "Expected a distribution")
      local d = assert(v.obj, "Needs to call sample before")
      local w = normalize_shape(d, w)
      local wgrads = normalize_shape(d, wgrads)
      d:logpdf_derivative(w, wgrads)
    end
  end
end

------------------------------------------------------------------------------

