learning_rate  = 1.0
momentum       = 0.1
weight_decay   = 1e-06
random1        = random(2134)
random2        = random(4283)
bunch_size     = 4
epsilon        = 1e-02

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
function check_result(trainer, filter, t, testname)
  if not filter then filter = function(x) return x end end
  local k=1
  assert(math.abs(trainer:validate_dataset(data) - t[k]) < epsilon,
	 "["..testname.."] Incorrect validation loss ")
  k=k+1
  for i = 1,ds_input:numPatterns() do
    value = filter(trainer:calculate(ds_input:getPattern(i)):toTable()[1])
    assert(math.abs(value - t[k]) < epsilon,
	   string.format("[%s] Incorrect result for input %s",
			 testname,
			 table.concat(ds_input:getPattern(i),",")))
    k=k+1
  end
  for name,cnn in trainer:iterate_weights() do
    local w = cnn:matrix():toTable()
    for _,v in ipairs(w) do
      assert(v-t[k]<epsilon,
	     string.format("[%s] Incorrect weight, expected %f, found %f",
			   name, t[k], v)) k=k+1 end
  end
end
-----------------------------------------------------------

-- All All Test => stack, dot product and bias
-- print("#######################################################")
-- print("# All All Test => stack, hyperplane and actf components#")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  trainer:train_dataset(data)
end
check_result(trainer, math.exp,
	     {
	       3.2876290788408e-05,
	       6.5825275946909e-05,
	       0.9999679818462,
	       0.99996795548262,
	       0.99999838598233,
	       1.4717608690262,
	       2.0912199020386,
	       0.014951737597585,
		 -3.2043704986572,
		 -3.2102212905884,
		 -4.3880376815796,
		 -4.3826746940613,
		 -5.3699712753296,
		 -8.0481376647949
	     },
	     "ALL ALL => STACK, HYPERPLANE, ACTF")

-- All All Test with replacement
--print("#######################################################")
--print("# All All Test  with replacement                      #")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
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
check_result(trainer, math.exp,
	     {
	       3.202018342563e-05,
	       6.297676236423e-05,
	       0.99996820502908,
	       0.99996846180768,
	       0.99999823089192,
		 -1.714160323143,
		 -1.9044058322906,
		 -0.057191781699657,
	       3.7159388065338,
	       3.7113511562347,
	       4.0735559463501,
	       4.0719871520996,
	       6.291428565979,
	       7.0797362327576,
	     },
	     "ALL ALL => WITH REPLACEMENT")

-- All All Test with distribution
--print("#######################################################")
--print("# All All Test  with distribution                     #")
net_component=ann.mlp.all_all.generate("2 inputs 2 tanh 1 log_logistic")
trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
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
check_result(trainer, math.exp,
	     {
	       3.3244676160393e-05,
	       6.6640150073464e-05,
	       0.99996755837365,
	       0.99996791787076,
	       0.99999818713814,
	       1.4724569320679,
		 -2.1051616668701,
		 -0.01775535941124,
		 -3.241943359375,
		 -3.2435941696167,
	       4.443178653717,
	       4.4328060150146,
		 -5.3502349853516,
	       7.977089881897,
	     },
	     "ALL ALL => WITH DISTRIBUTION")

-- Join and Copy Test => stack, join, copy, hyperplane and actf, components
--print("#######################################################")
--print("# Join and Copy Test => stack, join, copy, hyperplane and actf, components #")

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

trainer=trainable.supervised_trainer(net_component,
				     ann.loss.cross_entropy(1),
				     bunch_size)
trainer:set_option("learning_rate", learning_rate)
trainer:set_option("momentum",      momentum)
trainer:set_option("weight_decay",  weight_decay)
trainer:build()
trainer:randomize_weights{
  random = random1,
  inf    = -0.1,
  sup    = 0.1
}
for i=1,10000 do
  trainer:train_dataset(data)
end
check_result(trainer, math.exp,
	     {
	       4.4288186472841e-05,
	       9.7439813021598e-05,
	       0.99996001794975,
	       0.99996027395866,
	       0.99999999976159,
		 -1.214913725853,
	       1.527724981308,
		 -3.9111864566803,
	       2.4031643867493,
	       2.4322991371155,
		 -3.0390210151672,
		 -3.0553390979767,
	       3.8532137870789,
		 -5.0288019180298,
	       8.7805194854736,
	       8.7100381851196,
	     },
	     "JOIN AND COPY")

-- TEST ACTIVATION FUNCTIONS

test_tables = {
  {
    2.5464187274338e-05,
    5.1619176485562e-05,
    0.99998226146543,
    0.99998226085608,
    0.99998524083435,
      -2.1771562099457,
    2.1249141693115,
      -0.50905787944794,
      -0.35597959160805,
    0.52946943044662,
    5.5774660110474,
    5.6048789024353,
      -5.4532189369202,
      -5.4271574020386,
    6.0401663780212,
      -5.9397482872009,
    4.4949231147766,
      -4.3562297821045,
    6.5617160797119,
    5.2226033210754,
  },
  {
    5.7184870456695e-06,
    1.8896271961548e-05,
    0.9999980085169,
    0.99999801376309,
    1,
      -0.20611572265625,
      -0.14534106850624,
      -3.1036338806152,
    2.185019493103,
      -4.5230655670166,
    3.0648052692413,
    3.0568759441376,
    2.327056646347,
    2.341545343399,
    2.9545421600342,
    2.0297336578369,
    0.44760897755623,
    0.69581699371338,
    3.1761589050293,
      -2.8681812286377,
  },
  {
    2.3777005480952e-05,
    4.798930451884e-05,
    0.99998428885893,
    0.99998428888985,
    0.99998430410546,
      -0.83395653963089,
      -0.077860958874226,
      -0.018816085532308,
      -0.018558878451586,
    0.55860668420792,
    1.6126940250397,
    1.6157685518265,
      -0.26697292923927,
      -0.24897348880768,
    2.1173193454742,
      -0.2741239964962,
    2.1042129993439,
      -0.27080279588699,
    6.3520913124084,
    4.1515502929688,
  },
  {
    0.0001169169700006,
    0.0002340627174136,
    0.99988279533913,
    0.99988364410415,
    0.99999999173097,
    1.0129239559174,
      -0.027087911963463,
    1.0524189472198,
      -0.072163484990597,
    1.1806898117065,
      -0.96557080745697,
      -0.96798765659332,
    0.035413827747107,
      -0.047438681125641,
      -1.051481962204,
    0.013136994093657,
    1.5490934848785,
      -0.054378807544708,
    7.8850965499878,
      -9.5449600219727,
  },
  {
    3.1247047900251e-06,
    2.9644476079416e-06,
    0.99999875034506,
    0.99999875076206,
    1,
    0.9813414812088,
    0.046443276107311,
    0.54657065868378,
    0.056455112993717,
    0.41570544242859,
      -2.283091545105,
      -2.288688659668,
      -0.18083891272545,
      -0.07964999973774,
    3.2540953159332,
    0.16385380923748,
    0.8683745265007,
    0.11332588642836,
      -3.2964270114899,
      -0.86549544334412,
  },
}

for idx,actf in ipairs({"softsign", "softplus", "sin", "hardtanh", "linear"}) do
  --print("#######################################################")
  --printf("# %s Test                                       #\n", actf)
  net_component=ann.mlp.all_all.generate(string.format("2 inputs 2 %s 2 %s 1 log_logistic",
						       actf, actf))
  trainer=trainable.supervised_trainer(net_component,
				       ann.loss.cross_entropy(1),
				       bunch_size)
  trainer:set_option("learning_rate", learning_rate)
  trainer:set_option("momentum",      momentum)
  trainer:set_option("weight_decay",  weight_decay)
  trainer:build()
  trainer:randomize_weights{
    random = random1,
    inf    = -0.1,
    sup    = 0.1
  }
  for i=1,10000 do
    trainer:train_dataset(data)
  end
  check_result(trainer, math.exp,
	       test_tables[idx],
	       "["..actf.."]")
end
