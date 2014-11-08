 -- forces CUDA when available
mathcore.set_use_cuda_default(util.is_cuda_available())
--
local bunch_size = 4
local thenet  = ann.mlp.all_all.generate("2 inputs 2 logistic 1 logistic")
local trainer = trainable.supervised_trainer(thenet, ann.loss.mse(), bunch_size)
trainer:build()
trainer:randomize_weights{
  random     =  random(1234),
  inf        = -0.1,
  sup        =  0.1 }

trainer:set_option("learning_rate", 8.0)
trainer:set_option("momentum",      0.8)

local m_xor = matrix.fromString[[
    4 3
    ascii
    0 0 0
    0 1 1
    1 0 1
    1 1 0
]]

local ds_input  = dataset.matrix(m_xor, {patternSize={1,2}})
local ds_output = dataset.matrix(m_xor, {offset={0,2}, patternSize={1,1}})

local rnd=random(523)
for i=1,10000 do  
  local tr_error = trainer:train_dataset{ input_dataset  = ds_input,
					  output_dataset = ds_output,
					  shuffle        = rnd }
  print(i, tr_error)
end
