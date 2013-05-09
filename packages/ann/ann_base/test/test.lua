learning_rate  = 1.0
momentum       = 0.1
weight_decay   = 1e-06
random1        = random(2134)
random2        = random(4283)
bunch_size     = 4

-----------------------------------------------------------
m_or = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 1
]]
ds_input  = dataset.matrix(m_or,{patternSize={1,2}})
ds_output = dataset.matrix(m_or,{offset={0,2},patternSize={1,1}})
data = {
  input_dataset  = ds_input,
  output_dataset = ds_output,
  shuffle        = random2
}
function show_weights(trainer, filter)
  if not filter then filter = function(x) return x end end
  print()
  for i = 1,ds_input:numPatterns() do
    value = filter(trainer:calculate(ds_input:getPattern(i))[1])
    printf("%s\t %s\n",
	   table.concat(ds_input:getPattern(i),","),
	   value)
  end
  print()
  trainer:show_weights()
end
-----------------------------------------------------------

-- All All Test => stack, dot product and bias
print("#######################################################")
print("# All All Test => stack, hyperplane and actf components#")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
net_component:set_option("learning_rate", learning_rate)
net_component:set_option("momentum",      momentum)
net_component:set_option("weight_decay",  weight_decay)
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  trainer:train_dataset(data)
end
print(trainer:validate_dataset(data))
show_weights(trainer, math.exp)

-- All All Test with replacement
print("#######################################################")
print("# All All Test  with replacement                      #")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
net_component:set_option("learning_rate", learning_rate)
net_component:set_option("momentum",      momentum)
net_component:set_option("weight_decay",  weight_decay)
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  err = trainer:train_dataset{
    input_dataset  = data.input_dataset,
    output_dataset = data.output_dataset,
    shuffle        = data.shuffle,
    replacement    = 4,
  }
end
print(trainer:validate_dataset(data))
show_weights(trainer, math.exp)

-- All All Test with distribution
print("#######################################################")
print("# All All Test  with distribution                     #")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
net_component:set_option("learning_rate", learning_rate)
net_component:set_option("momentum",      momentum)
net_component:set_option("weight_decay",  weight_decay)
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  err = trainer:train_dataset{
    distribution = { { input_dataset  = data.input_dataset,
		       output_dataset = data.output_dataset,
		       probability    = 0.3, },
		     { input_dataset  = data.input_dataset,
		       output_dataset = data.output_dataset,
		       probability    = 0.7, },
    },
    shuffle     = data.shuffle,
    replacement = 4,
  }
end
print(trainer:validate_dataset(data))
show_weights(trainer, math.exp)

-- Join and Copy Test => stack, join, copy, hyperplane and actf, components
print("#######################################################")
print("# Join and Copy Test => stack, join, copy, hyperplane and actf, components #")

net_component=ann.components.stack()
net_component:push( ann.components.copy{ times=2, input=2 } )
join=ann.components.join() net_component:push( join )
h1 = ann.components.stack()
h1:push( ann.components.hyperplane{ input=2, output=2 } )
h1:push( ann.components.actf.tanh() )
join:add( h1 )
join:add( ann.components.base{ size=2 } )
net_component:push( ann.components.hyperplane{ input=4, output=1 })
net_component:push( ann.components.actf.log_logistic() )

net_component:set_option("learning_rate", learning_rate)
net_component:set_option("momentum",      momentum)
net_component:set_option("weight_decay",  weight_decay)
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  trainer:train_dataset(data)
end
print(trainer:validate_dataset(data))
show_weights(trainer, math.exp)

-- TEST ACTIVATION FUNCTIONS

for _,actf in ipairs({"softsign", "softplus", "sin", "hardtanh", "linear"}) do
  print("#######################################################")
  printf("# %s Test                                       #\n", actf)
  net_component=ann.mlp.all_all.generate(string.format("2 inputs 2 %s 2 %s 1 log_logistic",
						       actf, actf))
  net_component:set_option("learning_rate", learning_rate)
  net_component:set_option("momentum",      momentum)
  net_component:set_option("weight_decay",  weight_decay)
  trainer=trainable.supervised_trainer(net_component,
				       ann.loss.cross_entropy(1),
				       bunch_size)
  trainer:build()
  trainer:randomize_weights{
    random = random1,
    inf    = -0.1,
    sup    = 0.1
  }
  for i=1,10000 do
    trainer:train_dataset(data)
  end
  print(trainer:validate_dataset(data))
  show_weights(trainer, math.exp)
end
