get_table_from_dotted_string("bayesian", true)

local wrap_matrices = matrix.dict.wrap_matrices

function bayesian.validate_dataset(eval, t)
  local params = get_table_fields(
    {
      -- In this case, we check all the given parameters, because all the
      -- dataset iteration schemes are not available for validate_dataset
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = true },
      bunch_size     = { type_match = "number", mandatory = true },
      rnd            = { isa_match  = random, mandatory = false, default=nil },
      replacement    = { type_match = "number", mandatory = false, default=nil },
      samples        = { type_match = "table", mandatory = true },
      loss           = { isa_match = ann.loss, mandatory = true },
      N              = { type_match = "number", mandatory = true, default=100 },
    }, t, true)
  local samples  = params.samples
  local rnd      = params.rnd
  local loss     = params.loss
  local N        = params.N
  local invN     = 1.0/N
  params.samples = nil
  params.rnd     = nil
  params.loss    = nil
  params.N       = nil
  if params.replacement then params.shuffle = rnd end
  assert(#samples > 0, "Samples table is empty")
  loss:reset()
  for input,target in trainable.dataset_pair_iterator(params) do
    local output = matrix.as(target:get_matrix()):zeros()
    for i=1,N do
      local which = rnd:choose(samples)
      local out = eval(which, input)
      assert(isa(out, matrix), "Eval function must return a matrix")
      output:axpy(invN, out)
    end
    loss:accum_loss(loss:compute_loss(output,target))
  end
  return loss:get_accum_loss()
end
