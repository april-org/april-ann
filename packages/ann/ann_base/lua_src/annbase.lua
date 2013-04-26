ann = ann or {}
ann.stopping_criterions = ann.stopping_criterions or {}

function ann.stopping_criterions.make_max_epochs_wo_imp_absolute(abs_max)
  local f = function(params)
    return (params.current_epoch - params.best_epoch) >= abs_max
  end
  return f
end

function ann.stopping_criterions.make_max_epochs_wo_imp_relative(rel_max)
  local f = function(params)
    return not (params.current_epoch/params.best_epoch < rel_max)
  end
  return f
end

-----------------------------------------------------------------------------

local function check_train_crossvalidation_params(params)
  local check = table.invert{ "ann", "training_table", "validation_table",
			      "max_epochs", "stopping_criterion",
			      "validation_function",
			      "update_function", "first_epoch", "min_epochs" }
  for name,value in pairs(params) do
    if not check[name] then error ("Incorrect param name: " .. name) end
  end
end

local function check_train_wo_validation_params(params)
  local check = table.invert{ "ann", "training_table", "update_function",
			      "max_epochs", "min_epochs",
			      "percentage_stopping_criterion" }
  for name,value in pairs(params) do
    if not check[name] then error ("Incorrect param name: " .. name) end
  end
end

-- params is a table like this:
-- { ann = neural_network,
--   training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   validation_table = { input_dataset = ...., output_dataset = ....., .....},
--   validation_function = function( thenet, validation_table ) .... end
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   -- train_params is this table ;)
--   stopping_criterion = function{ current_epoch, best_epoch, best_val_error, train_error, validation_error, train_params } .... return true or false end
--   update_function = function{ current_epoch, best_epoch, best_val_error, 
--   first_epoch = 1
-- }
--
-- and returns this:
--  return { best_net         = best_net,
--	     best_epoch       = best_epoch,
--	     best_val_error   = best_val_error,
--	     num_epochs       = epoch,
--	     last_train_error = last_train_error,
--	     last_val_error   = last_val_error }
function ann.train_crossvalidation(params)
  check_train_crossvalidation_params(params)
  params.first_epoch = params.first_epoch or 1
  if not params.max_epochs then error("Needs max_epochs field") end
  if not params.ann then error("Needs ann field") end
  if not params.training_table then error("Needs training_table field") end
  if not params.validation_table then error("Needs validation_table field") end
  if not params.stopping_criterion then error("Needs stopping_criterion field") end
  if not params.min_epochs then error("Needs a min_epochs field") end
  params.update_function = params.update_function or function(t) return end
  if not params.validation_function then
    params.validation_function = function(thenet, t)
      return thenet:validate_dataset(t)
    end
  end
  local thenet           = params.ann
  local best_epoch       = 1
  local best_net         = thenet:clone()
  local best_val_error   = params.validation_function(thenet,
						      params.validation_table)
  local last_val_error   = best_val_error
  local last_train_error = 0
  local last_epoch       = 0
  for epoch=params.first_epoch,params.max_epochs do
    collectgarbage("collect")
    clock = util.stopwatch()
    clock:go()

    local tr_error  = thenet:train_dataset(params.training_table)
    local val_error = params.validation_function(thenet, params.validation_table)
    last_train_error,last_val_error,last_epoch = tr_error,val_error,epoch

    clock:stop()
    cpu, wall = clock:read()

    if val_error < best_val_error then
      best_epoch     = epoch
      best_val_error = val_error
      best_net       = thenet:clone()
    end

    
    params.update_function({ current_epoch    = epoch,
			     best_epoch       = best_epoch,
			     best_val_error   = best_val_error,
			     train_error      = tr_error,
			     validation_error = val_error,
           cpu              = cpu,
           wall             = wall,
			     train_params     = params })
    if (epoch > params.min_epochs and
	params.stopping_criterion({ current_epoch    = epoch,
				    best_epoch       = best_epoch,
				    best_val_error   = best_val_error,
				    train_error      = tr_error,
				    validation_error = val_error,
				    train_params     = params })) then
      break						  
    end
  end
  return { best_net         = best_net,
	   best_val_error   = best_val_error,
	   best_epoch       = best_epoch,
	   last_epoch       = last_epoch,
	   last_train_error = last_train_error,
	   last_val_error   = last_val_error }
end

-- This function trains the ANN without validation, it is trained until a
-- maximum of epochs or until the improvement in training error is less than
-- given percentage
--
-- params is a table like this:
-- { ann = neural_network,
--   training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   min_epochs = NUMBER,
--   max_epochs = NUMBER,
--   update_function = function{ current_epoch, best_epoch, best_val_error,
--   percentage_stopping_criterion = NUMBER (normally 0.01 or 0.001)
-- }
--
-- returns the trained ANN 
function ann.train_wo_validation(params)
  check_train_wo_validation_params(params)
  if not params.max_epochs then error("Needs max_epochs field") end
  if not params.ann then error("Needs ann field") end
  if not params.training_table then error("Needs training_table field") end
  if not params.min_epochs then error("Needs a min_epochs field") end
  if not params.percentage_stopping_criterion then
    error("Needs a percentage_stopping_criterion field")
  end
  params.update_function = params.update_function or function(t) return end
  local thenet      = params.ann
  local prev_tr_err = 111111111
  local best_net    = thenet:clone()
  for epoch=1,params.max_epochs do
    local tr_table = params.training_table
    if type(tr_table) == "function" then tr_table = tr_table() end
    collectgarbage("collect")
    local tr_err         = thenet:train_dataset(tr_table)
    local tr_improvement = (prev_tr_err - tr_err)/prev_tr_err
    if (epoch > params.min_epochs and
	tr_improvement < params.percentage_stopping_criterion) then
      break
    end
    best_net = thenet:clone()
    params.update_function{ current_epoch     = epoch,
			    train_error       = tr_err,
			    train_improvement = tr_improvement,
			    train_params      = params }
    prev_tr_err = tr_err
  end
  return best_net
end

------------------------------------------------------------------------

ann.mlp         = ann.mlp or {}
ann.mlp.all_all = ann.mlp.all_all or {}
function ann.mlp.all_all.generate(topology)
  local thenet = ann.components.stack()
  local name   = "layer"
  local count  = 1
  local t      = string.tokenize(topology)
  local prev_size = tonumber(t[1])
  for i=3,#t,2 do
    local size = tonumber(t[i])
    local actf = t[i+1]
    thenet:push( ann.components.hyperplane{
		   input=prev_size, output=size,
		   bias_weights="b" .. count,
		   dot_product_weights="w" .. count,
		   name="layer" .. count,
		   bias_name="b" .. count,
		   dot_product_name="w" .. count } )
    if not ann.components[actf] then
      error("Incorrect activation function: " .. actf)
    end
    thenet:push( ann.components[actf]() )
    count = count + 1
  end
  return thenet
end
