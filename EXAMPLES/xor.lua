thenet = ann.mlp.all_all.generate{
  topology   =  "2 inputs 2 logistic 1 logistic",
  random     =  random(1234),
  use_fanin  =  false,
  inf        = -0.1,
  sup        =  0.1,
  bunch_size =  1 }

thenet:set_option("learning_rate", 1.0)
thenet:set_option("momentum",      0.5)
thenet:set_option("weight_decay",  1e-05)

m_xor = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 0
]]

ds_input  = dataset.matrix(m_xor, {patternSize={1,2}})
ds_output = dataset.matrix(m_xor, {offset={0,2}, patternSize={1,1}})

rnd=random(523)
for i=1,10000 do
  local error = thenet:train_dataset{ input_dataset  = ds_input,
				      output_dataset = ds_output,
				      shuffle        = rnd }
  print(i, error)
end
