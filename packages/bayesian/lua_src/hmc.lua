get_table_from_dotted_string("bayesian", true)

local wrap_matrices = matrix.dict.wrap_matrices

-- Hamiltonian Monte-Carlo implementation with a basic on-line adaptation of
-- epsilon, bounded by [epsilon_min,epsilon_max].


-- Radford M. Neal. MCMC using Hamiltonian dynamics. 2010
-- http://www.cs.toronto.edu/~radford/ham-mcmc.abstract.html

-- Mathew D. Hoffman, Andrew Gelman. The No-U-Turn Sampler: Adaptively Setting
-- Path Lengths in Hamiltonian Monte Carlo. JMLR 2014.
-- http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf

-- http://deeplearning.net/tutorial/hmc.html

-- @param eval is a function a function which returns L(Theta),dL/dTheta. It is
-- implemented to minimize the negative of the log-likelihood (maximize the
-- log-likelihood).
--
-- @param theta is a matrix, a table of matrices or a matrix.dict instance.
local function hmc(self, eval, theta)
  local state       = self.state
  --
  local energies    = state.energies
  local math_log    = math.log
  local math_clamp  = math.clamp
  local priors      = state.priors
  local samples     = state.samples
  local theta       = wrap_matrices(theta)
  --
  local acc_decay   = self:get_option("acc_decay")
  local alpha       = self:get_option("alpha")
  local epsilon     = state.epsilon or self:get_option("epsilon")
  local epsilon_dec = self:get_option("epsilon_dec")
  local epsilon_inc = self:get_option("epsilon_inc")
  local epsilon_max = self:get_option("epsilon_max")
  local epsilon_min = self:get_option("epsilon_min")
  local nsteps      = self:get_option("nsteps")
  local rng         = state.rng or random(self:get_option("seed"))
  local scale       = self:get_option("scale")
  local thin        = self:get_option("thin")
  local mass        = self:get_option("mass")
  local persistence  = self:get_option("persistence")
  local target_acceptance_rate = self:get_option("target_acceptance_rate")
  --
  assert(persistence >= 0.0 and persistence <= 1.0,
         "Incorrect persistence ratio")
  local inv_mass     = 1.0/mass
  local spersistence = math.sqrt(1 - persistence*persistence)
  state.acceptance_rate = state.acceptance_rate
  --
  -- kinetic energy associated with given velocity
  local kinetic_energy = function(vel)
    return 0.5 * vel:dot(vel)
  end
  --
  -- executes the simulation chain of HMC using leapfrog updates
  local simulation = function(pos, vel, epsilon, nsteps)
    --
    -- receives the optimizer table, a matrix dict with positions, other with
    -- velocities, and the epsilon for the step
    local leapfrog = function(pos, vel, epsilon, i)
      -- from pos(t) and vel(t - eps/2), compute vel(t + eps/2)
      local _,grads = eval(i)
      grads = wrap_matrices(grads)
      vel:axpy(-epsilon, grads)
      -- from vel(t + eps/2) compute pos(t + eps)
      pos:axpy(epsilon*inv_mass, vel)
      -- local eval_result,grads = eval()
      -- vel:axpy(-epsilon*0.5, grads)
    end
    --
    -- compute velocity at time: t + eps/2
    local initial_energy,grads = eval(0)
    initial_energy = scale*initial_energy + priors:compute_neg_log_prior(pos)
    grads = wrap_matrices(grads)
    vel:axpy(-0.5*epsilon, grads)
    -- compute position at time: t + eps
    pos:axpy(epsilon*inv_mass, vel)
    -- compute from 2 to nsteps leapfrog updates
    for i=2,nsteps do
      leapfrog(pos, vel, epsilon, i-1)
    end
    -- compute velocity at time: t + nsteps*eps
    local final_energy,grads = eval(nsteps)
    final_energy = scale*final_energy + priors:compute_neg_log_prior(pos)
    grads = wrap_matrices(grads)
    vel:axpy(-0.5*epsilon, grads)
    return initial_energy, final_energy
  end
  --
  -- metropolis hastings reject procedure
  local metropolis_hastings = function(initial_p, final_p)
    local alpha = initial_p - final_p
    local accept_threshold = math_log(rng:randDblExc())
    return accept_threshold < alpha
  end
  --
  -- priors sampling
  priors:sample(rng, theta)
  --
  -- one HMC sample procedure
  local norm01 = stats.dist.normal()
  local theta0 = theta:clone() -- for in case of rejection
  local vel    = self.state.vel
  local vel0 -- only if persistent
  -- sample velocity from a standard normal distribution
  if persistence == 0.0 or not vel then
    vel = vel or theta:clone_only_dims()
    for name,v in pairs(vel) do
      norm01:sample(rng, v:rewrap(v:size(), 1))
    end
  else
    vel0 = vel:clone() -- for in case of rejection
    vel:scal(-persistence)
    for name,v in pairs(vel) do
      local aux = matrix.col_major(v:size(),1)
      norm01:sample(rng, aux)
      v:rewrap(v:size(),1):axpy(spersistence, aux)
    end
  end
  -- epsilon perturbation
  local p_epsilon = epsilon + rng:randNorm(0,alpha*alpha)
  local initial_kinetic = kinetic_energy(vel) * inv_mass
  -- simulate the HMC mechanics
  local initial_energy, final_energy = simulation(theta, vel, p_epsilon, nsteps)
  vel:scal(-1.0)
  local final_kinetic = kinetic_energy(vel) * inv_mass
  -- rejection based in metropolis hastings
  local accept = metropolis_hastings(initial_energy + initial_kinetic,
                                     final_energy + final_kinetic)
  --
  local energy = final_energy
  local ok =  pcall(theta.prune_subnormal_and_check_normal, theta)
  -- if not ok then print(ok, "PROBLEM") end
  if not accept or not ok then
    energy = initial_energy
    theta:copy(theta0)
    if persistent then vel:copy(vel0) end
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
  state.initial_kinetic = initial_kinetic
  state.final_kinetic = final_kinetic
  state.epsilon = epsilon
  state.rng = rng
  state.vel = vel
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
  return eval()
end

local hmc_methods, hmc_class_metatable = class("bayesian.optimizer.hmc",
                                               ann.optimizer)

function hmc_class_metatable:__call(g_options, l_options, count, state)
  -- the base optimizer, with the supported learning parameters
  local obj = ann.optimizer({
                              { "thin", "Take 1-of-thin samples (1)" },
                              { "acc_decay", "Acceptance average mean decay (0.95)" },
                              { "target_acceptance_rate", "Desired acceptance rate (0.65)" },
                              { "alpha", "Step length perturbation stddev (0.1)" },
                              { "mass", "Mass of particles (1)" },
                              { "nsteps", "Number of Leap-Frog steps (20)" },
                              { "epsilon", "Initial epsilon value (0.01)" },
                              { "epsilon_inc", "Epsilon increment (1.02)" },
                              { "epsilon_dec", "Epsilon decrement (0.98)" },
                              { "epsilon_min", "Epsilon lower bound (1e-04)" },
                              { "epsilon_max", "Epsilon upper bound (1.0)" },
                              { "persistence", "Momenta persistence, 0 for non-persistent (0)" },
                              { "scale", "Energy function scale (1)" },
                              { "seed", "Seed for random number generator (time)" },
                            },
                            g_options,
                            l_options,
                            count)
  obj.state = state or
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
  obj = class_instance(obj, self)
  obj:set_option("thin", 1)
  obj:set_option("acc_decay", 0.95)
  obj:set_option("target_acceptance_rate", 0.65)
  obj:set_option("alpha", 0.1)
  obj:set_option("mass", 1)
  obj:set_option("nsteps", 20)
  obj:set_option("epsilon", 0.01)
  obj:set_option("epsilon_inc", 1.02)
  obj:set_option("epsilon_dec", 0.98)
  obj:set_option("epsilon_min", 1e-04)
  obj:set_option("epsilon_max", 1.0)
  obj:set_option("persistence", 0.0)
  obj:set_option("scale", 1.0)
  return obj
end

hmc_methods.execute = hmc

function hmc_methods:clone()
  local obj = bayesian.optimizer.hmc()
  obj.count             = self.count
  obj.layerwise_options = table.deep_copy(self.layerwise_options)
  obj.global_options    = table.deep_copy(self.global_options)
  for k,v in pairs(self.state) do obj.state[k] = v end
  for k,v in pairs(self.state.samples) do obj.state.samples[k] = v:clone() end
  return obj
end

function hmc_methods:to_lua_string(format)
  local format = format or "binary"
  local str_t = { "ann.optimizer.hmc(",
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

function hmc_methods:start_burnin()
  self.state.samples  = {}
  self.state.energies = {}
end

function hmc_methods:finish_burnin()
  self.state.samples  = {}
  self.state.energies = {}
end

function hmc_methods:get_samples()
  return self.state.samples
end

function hmc_methods:get_state_string()
  return "%6d %11g :: %6d %11g %6.2f%% %s"%
  {
    self:get_count(), self.state.energy, #self.state.samples,
    self.state.epsilon,
    self.state.acceptance_rate*100, (self.state.accepted and "**") or ""
  }
end

function hmc_methods:get_state_table()
  return self.state
end

function hmc_methods:get_priors()
  return self.state.priors
end
