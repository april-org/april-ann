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

local function check_train_crossvalidation_params(params)
  local check = table.invert{ "ann", "training_table", "validation_table",
			      "max_epochs", "stopping_criterion",
			      "update_function", "first_epoch" }
  for name,value in pairs(params) do
    if not check[name] then error ("Incorrect param name: " .. name) end
  end
end

-- params is a table like this:
-- { ann = neural_network,
--   training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   validation_table = { input_dataset = ...., output_dataset = ....., .....},
--   validation_func = function( thenet, validation_table ) .... end
--   max_epochs = NUMBER,
--   -- train_params is this table ;)
--   stopping_criterion = function{ current_epoch, best_epoch, best_val_error, train_error, validation_error, train_params } .... return true or false end
--   update_function = function{ current_epoch, best_epoch, best_val_error, 
--   first_epoch = 1
-- }
--
-- and returns this:
-- { best_net = best_net, best_epoch = best_epoch,
--   best_val_error = best_val_error }
function ann.train_crossvalidation(params)
  check_train_crossvalidation_params(params)
  params.first_epoch = params.first_epoch or 1
  if not params.max_epochs then error("Needs max_epochs field") end
  if not params.ann then error("Needs ann field") end
  if not params.training_table then error("Needs training_table field") end
  if not params.validation_table then error("Needs validation_table field") end
  if not params.stopping_criterion then error("Needs stopping_criterion field") end
  params.update_function = params.update_function or function(t) return end
  local thenet         = params.ann
  local best_epoch     = 0
  local best_net       = thenet:clone()
  local best_val_error = thenet:validate_dataset(params.validation_table)
  params.ann = nil
  if not params.validation_func then
    params.validation_func = function(thenet, t)
      return thenet:validate_dataset(t)
    end
  end
  for epoch=params.first_epoch,params.max_epochs do
    collectgarbage("collect")
    local tr_error  = thenet:train_dataset(params.training_table)
    local val_error = safe_call(params.validation_func, {},
				thenet, params.validation_table)
    if val_error < best_val_error then
      best_epoch     = epoch
      best_val_error = val_error
      best_net       = thenet:clone()
    end
    safe_call(params.update_function, {},
	      { current_epoch    = epoch,
		best_epoch       = best_epoch,
		best_val_error   = best_val_error,
		train_error      = tr_error,
		validation_error = val_error,
		train_params     = params })
    if safe_call(params.stopping_criterion, {},
		 { current_epoch    = epoch,
		   best_epoch       = best_epoch,
		   best_val_error   = best_val_error,
		   train_error      = tr_error,
		   validation_error = val_error,
		   train_params     = params }) then
      break						  
    end
  end
  return { best_net = best_net, best_epoch = best_epoch,
	   best_val_error = best_val_error }
end
