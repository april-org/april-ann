get_table_from_dotted_string("bayesian", true)

local wrap_matrices = matrix.dict.wrap_matrices

-- modifies model weights to be the MAP model for a given eval function
function bayesian.build_MAP(model, eval, samples)
  assert(samples and energies, "Needs a table with samples and energies fields")
  assert(#samples > 0, "Tables are empty")
  local best,argbest = eval(samples[1]),samples[1]
  for i=2,#samples do
    if energies[i] < best then
      best,argbest = eval(samples[1]),samples[i]
    end
  end
  model:build{ weights = best:clone() }
  return model
end

-- returns a model which forwards computation is a combination of N sampled
-- parameters
function bayesian.build_bayes_comb(t)
  local params = get_table_fields(
    {
      forward = { mandatory = true, type_match = "function" },
      shuffle = { isa_match = random, mandatory = false, default=nil },
      samples = { type_match = "table", mandatory = true },
      N       = { type_match = "number", mandatory = true, default=100 },
    }, t, true)  
  assert(#params.samples > 0, "samples table is empty")
  return ann.components.wrapper{
    state = {
      N       = params.N,
      samples = params.samples,
      rnd     = params.shuffle,
      forward = params.forward,
    },
    weights = params.samples[#params.samples],
    forward = function(self,input)
      local forward = self.state.forward
      local N       = self.state.N
      local rnd     = self.state.rnd
      local samples = self.state.samples
      local invN    = 1/N
      local which   = rnd:choose(samples)
      local out     = forward(which, input)
      if isa(out, tokens.base) then out = out:get_matrix() end
      assert(isa(out, matrix), "The forward function must return a matrix")
      local output  = out:clone():scal(invN)
      for i=2,N do
        local which = rnd:choose(samples)
        local out   = forward(which, input)
        if isa(out, tokens.base) then out = out:get_matrix() end
        output:axpy(invN, out)
      end
      return output
    end,
  }
end
