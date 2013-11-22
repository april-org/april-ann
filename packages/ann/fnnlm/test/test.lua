nnlm = ann.fnnlm{
  factors = {
    { name="c", layers={ {size=74},{size=20, actf="softplus"} }, order=3 },
    { name="w", layers={ {size=1700},{size=64, actf="softplus"} }, order=3 },
  },
  output_size=128,
  hidden_actf="logistic",
  hidden_size=32,
}

trainer = nnlm:get_trainer()
trainer:build()
trainer:randomize_weights{
  inf = -0.1,
  sup =  0.1,
  random = random(1234),
}

nnlm:set_option("learning_rate", 0.1)

output = nnlm:forward{
  {
    1, 2, 3,
    4, 96, 23,
  },
  {
    2, 3, 5,
    96, 23, 17,
  }
}

print(trainer:component("factors_join"):get_output():get_matrix())
print(output:get_matrix())
