get_table_from_dotted_string("bayesian.optimizer", true)

local wrap_matrices = matrix.dict.wrap_matrices

-- Random walk Metropolis update with normal proposals

-- @param eval is a function a function which returns L(Theta). It is
-- implemented to minimize the negative of the log-likelihood (maximize the
-- log-likelihood).
--
-- @param theta is a matrix, a table of matrices or a matrix.dict instance.
local function metropolis(self, eval, theta)
  local state       = self.state
  --
  local energies    = state.energies
  local math_log    = math.log
  local math_clamp  = math.clamp
  local priors      = state.priors
  local samples     = state.samples
  local origw = theta
  local theta = wrap_matrices(theta)
  --
  local acc_decay   = self:get_option("acc_decay")
  local epsilon     = state.epsilon or self:get_option("epsilon")
  local epsilon_dec = self:get_option("epsilon_dec")
  local epsilon_inc = self:get_option("epsilon_inc")
  local epsilon_max = self:get_option("epsilon_max")
  local epsilon_min = self:get_option("epsilon_min")
  local rng         = state.rng or random(self:get_option("seed"))
  local scale       = self:get_option("scale")
  local thin        = self:get_option("thin")
  local target_acceptance_rate = self:get_option("target_acceptance_rate")
  --
  -- metropolis hastings reject procedure
  local metropolis_hastings = function(initial_p, final_p)
    local alpha = initial_p - final_p
    local accept_threshold = math_log(rng:randDblExc())
    return accept_threshold < alpha
  end
  --
  -- one METROPOLIS sample procedure
  local norm01 = state.norm01 or stats.dist.normal()
  local theta0 = theta:clone() -- for in case of rejection
  local eval0_result = table.pack( eval(origw, 0) )
  local initial_energy = scale*eval0_result[1]
  for name,v in pairs(theta) do
    local aux = matrix.as(v)
    norm01:sample(rng, aux:rewrap(v:size(),1))
    v:axpy(epsilon, aux)
  end
  local eval1_result = table.pack( eval(origw, 1) )
  local final_energy = scale*eval1_result[1]
  -- rejection based in metropolis hastings
  local accept = metropolis_hastings(initial_energy, final_energy)
  --
  local result = eval1_result
  local energy = final_energy
  local ok =  pcall(theta.prune_subnormal_and_check_normal, theta)
  -- if not ok then print(ok, "PROBLEM") end
  if not accept or not ok then
    energy = initial_energy
    result = eval0_result
    theta:copy(theta0)
  end
  local accepted = (accept and 1) or 0
  -- accept rate update (exponential mean)
  assert(acc_decay > 0.0 and acc_decay < 1.0)
  --
  self:count_one()
  if self:get_count() % thin == 0 then
    table.insert(samples, theta:clone())
    table.insert(energies, energy)
  end
  local acceptance_rate
  if state.acceptance_rate then
    acceptance_rate = acc_decay * state.acceptance_rate + (1.0 - acc_decay) * accepted
  else
    acceptance_rate = accepted
  end
  -- sanity check
  assert(epsilon_inc > 1.0 and epsilon_dec < 1.0 and epsilon_dec > 0.0)
  -- depending in the acceptance_rate, the epsilon is updated to increase mixing
  -- if acc rate is greater than target, or to decrease mixing otherwise
  if acceptance_rate > target_acceptance_rate then
    epsilon = epsilon*epsilon_inc
  else
    epsilon = epsilon*epsilon_dec
  end
  epsilon = math_clamp(epsilon, epsilon_min, epsilon_max)
  --
  state.acceptance_rate = acceptance_rate
  state.accepted = accept
  state.energy = energy
  state.initial_energy = initial_energy
  state.final_energy = final_energy
  state.epsilon = epsilon
  state.rng = rng
  state.vel = vel
  state.norm01 = norm01
  --
  -- if #samples > samples_max_size then
  --   local next_samples = {}
  --   local permutation = rng:shuffle(#samples)
  --   for i=1,samples_max_size do
  --     -- table.insert(next_samples, rng:choose(samples))
  --     table.insert(next_samples, samples[permutation[i]])
  --   end
  --   samples = next_samples
  --   collectgarbage("collect")
  -- end
  -- print("\t\t ACCEPTED:", #samples, accepted, currentN, accepted/currentN)
  return table.unpack(eval0_result)
end

local metropolis_class,metropolis_methods = class("bayesian.optimizer.metropolis",
                                                  ann.optimizer)
bayesian.optimizer.metropolis = metropolis_class -- global environment

function metropolis_class:constructor(g_options, l_options, count, state)
  -- the base optimizer, with the supported learning parameters
  ann.optimizer.constructor(self,
                            {
                              { "thin", "Take 1-of-thin samples (1)" },
                              { "acc_decay", "Acceptance average mean decay (0.95)" },
                              { "target_acceptance_rate", "Desired acceptance rate (0.65)" },
                              { "epsilon", "Initial epsilon value (0.01)" },
                              { "epsilon_inc", "Epsilon increment (1.02)" },
                              { "epsilon_dec", "Epsilon decrement (0.98)" },
                              { "epsilon_min", "Epsilon lower bound (1e-04)" },
                              { "epsilon_max", "Epsilon upper bound (1.0)" },
                              { "scale", "Energy function scale (1)" },
                              { "seed", "Seed for random number generator (time)" },
                            },
                            g_options,
                            l_options,
                            count)
  self.state = state or
    {
      accepted        = false,
      acceptance_rate = nil,
      final_energy    = 0.0,
      final_kinetic   = 0.0,
      initial_energy  = 0.0,
      initial_kinetic = 0.0,
      energy = 0.0,
      epsilon = nil,
      priors = bayesian.priors(),
      samples = {},
      energies = {},
      rng = nil,
    }
  self:set_option("thin", 1)
  self:set_option("acc_decay", 0.95)
  self:set_option("epsilon", 0.1)
  self:set_option("epsilon_inc", 1.02)
  self:set_option("epsilon_dec", 0.98)
  self:set_option("epsilon_min", 1e-04)
  self:set_option("epsilon_max", 1.0)
  self:set_option("scale", 1)
  self:set_option("target_acceptance_rate", 0.40)
  return self
end

metropolis_methods.execute = metropolis

function metropolis_methods:clone()
  local obj = bayesian.optimizer.metropolis()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  for k,v in pairs(self.state) do obj.state[k] = v end
  for k,v in pairs(self.state.samples) do obj.state.samples[k] = v:clone() end
  return obj
end

function metropolis_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.metropolis(",
                  table.tostring(self.global_options),
                  ",",
                  table.tostring(self.layerwise_options),
                  ",",
                  tostring(self.count),
                  ",",
                  table.tostring(self.state),
                  ")" }
  return table.concat(str_t, "")
end

function metropolis_methods:start_burnin()
  self.state.samples  = {}
  self.state.energies = {}
end

function metropolis_methods:finish_burnin()
  self.state.samples  = {}
  self.state.energies = {}
end

function metropolis_methods:get_samples()
  return self.state.samples
end

function metropolis_methods:get_state_string()
  return "%6d %11g :: %6d %11g %6.2f%% %s"%
  {
    self:get_count(), self.state.energy, #self.state.samples,
    self.state.epsilon,
    self.state.acceptance_rate*100, (self.state.accepted and "**") or ""
  }
end

function metropolis_methods:get_state_table()
  return self.state
end

function metropolis_methods:get_priors()
  return self.state.priors
end

local metropolis_properties = {
}
function metropolis_methods:needs_property(name)
  return metropolis_properties[name]
end
