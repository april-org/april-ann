input_filename  = arg[1]
output_filename = arg[2]
bunch_size      = tonumber(arg[3] or 32)
thenet  = ann.mlp.all_all.load(input_filename)
trainer = trainable.supervised_trainer(thenet, nil, bunch_size)
trainer:save(output_filename)
