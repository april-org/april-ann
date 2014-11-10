get_table_from_dotted_string("bayesian", true)

-- modifies model weights to be the MAP model for a given eval function
function bayesian.get_MAP_weights(eval, samples)
  assert(samples, "Needs a table with samples as 2nd argument")
  assert(#samples > 0, "samples table is empty")
  local best,argbest = table.pack(eval(samples[1], 1)),samples[1]
  for i=2,#samples do
    local current = table.pack(eval(samples[i], i))
    if current[1] < best[1] then
      best,argbest = current,samples[i]
    end
  end
  return argbest,table.unpack(best)
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
      assert(class.is_a(out, matrix), "The forward function must return a matrix")
      local output  = out:clone():scal(invN)
      for i=2,N do
        local which = rnd:choose(samples)
        local out   = forward(which, input)
        output:axpy(invN, out)
      end
      return output
    end,
  }
end

------------------------------------------------------------------------------
