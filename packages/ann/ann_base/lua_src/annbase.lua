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
			      "update_function", "first_epoch", "min_epochs" }
  for name,value in pairs(params) do
    if not check[name] then error ("Incorrect param name: " .. name) end
  end
end

-- params is a table like this:
-- { ann = neural_network,
--   training_table   = { input_dataset = ...., output_dataset = ....., .....},
--   validation_table = { input_dataset = ...., output_dataset = ....., .....},
--   validation_func = function( thenet, validation_table ) .... end
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
  local thenet           = params.ann
  local best_epoch       = 1
  local best_net         = thenet:clone()
  local best_val_error   = thenet:validate_dataset(params.validation_table)
  local last_val_error   = best_val_error
  local last_train_error = 0
  local last_epoch       = 0
  if not params.validation_func then
    params.validation_func = function(thenet, t)
      return thenet:validate_dataset(t)
    end
  end
  for epoch=params.first_epoch,params.max_epochs do
    collectgarbage("collect")
    local tr_error  = thenet:train_dataset(params.training_table)
    local val_error = params.validation_func(thenet, params.validation_table)
    last_train_error,last_val_error,last_epoch = tr_error,val_error,epoch
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
