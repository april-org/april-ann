local thenet = ann.mlp.all_all.generate("256 inputs 128 tanh 10 log_softmax")
local thenet,cnns = thenet:build()
local loss = ann.loss.multi_class_cross_entropy()
local opt  = ann.optimizer.sgd()
opt:set_option("learning_rate", 0.01)
local rnd = random(1234)
for _,w in pairs(cnns) do w:uniformf(-0.1,0.1,rnd) end
local weight_grads = matrix.dict() -- upvalue for eval function
for i=1,1000 do
  local input  = matrix.col_major(1,256):uniformf(0,1,rnd)
  local target = matrix.col_major(1,10):zeros():set(1, rnd:randInt(1,10), 1.0)
  opt:execute(function(weights, it)
                if weights ~= cnns then
                  thenet:build{ weights = weights }
                end
                thenet:reset(it)
                local out = thenet:forward(input)
                local tr_loss,tr_matrix = loss:compute_loss(out,target)
                thenet:backprop(loss:gradient(out,target))
                weight_grads:zeros()
                weight_grads = thenet:compute_gradients(weight_grads)
                return tr_loss,weight_grads
              end,
              cnns)
end
