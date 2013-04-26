trainable = trainable or {}
trainable.supervised_trainer = trainable.supervised_trainer or {}

april_set_doc("trainable.supervised_trainer",
	      "Lua class for supervised machine learning models training")
class(trainable.supervised_trainer)

april_set_doc("trainable.supervised_trainer",
	      "Methods")
function trainable.supervised_trainer:__call(ann_component, loss_function)
  local obj = { ann_component = ann_component, loss_function = loss_function }
  setmetatable(obj, self)
  return obj
end

april_set_doc("trainable.supervised_trainer",
	      "\ttrain_step(input,target) => performs one training step "..
		"(reset, forward, loss, gradient, backprop, and update)")
function trainable.supervised_trainer:train_step(input, target)
  if type("input")  == "table" then input  = tokens.memblock(input)  end
  if type("target") == "table" then target = tokens.memblock(target) end
  self.ann_component:reset()
  self.loss_function:reset()
  local output   = self.ann_component:forward(input)
  local tr_loss  = self.loss_function:loss(output, target)
  local gradient = self.loss_function:gradient(output, target)
  self.ann_component:backprop(gradient)
  self.ann_component:update()
  return tr_loss,gradient
end

april_set_doc("trainable.supervised_trainer",
	      "\ttrain_step(t) => performs one training epoch with a given "..
		" table with datasets. Arguments:")
april_set_doc("trainable.supervised_trainer",
	      "\t                 t.input_dataset  dataset with input patterns")
april_set_doc("trainable.supervised_trainer",
	      "\t                 t.output_dataset  dataset with output patterns")
april_set_doc("trainable.supervised_trainer",
	      "\t                 t.bunch_size  mini batch size (bunch)")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.shuffle]  optional random object")
april_set_doc("trainable.supervised_trainer",
	      "\t                 [t.replacement]  optional replacement size")
function trainable.supervised_trainer:train_dataset(t)
  local params = get_table_fields(
    {
      input_dataset  = { mandatory = true },
      output_dataset = { mandatory = true },
      bunch_size     = { type_match = "number", mandatory = true },
      shuffle        = { type_match = "random", mandatroy = false, default=nil },
      replacement    = { type_match = "number", mandatroy = false, default=nil },
    }, t)
end
