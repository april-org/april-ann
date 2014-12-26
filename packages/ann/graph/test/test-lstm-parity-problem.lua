local CHECK_GRADIENTS = false
local BACKSTEP        = math.huge
local MAX_ERROR       = 0.04
local EPSILON         = 0.01
local MAX_SEQ_SIZE    = 10
local MAX_EPOCHS      = 1000 -- max epochs for sequence size = 2,MAX_SEQ_SIZE
local WEIGHT_DECAY    = 0.000001
local H               = 2 -- number of neurons in hidden layer

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local rnd1   = random(7576)
local rnd2   = random(1234)
local bern05 = stats.dist.bernoulli(0.5)
local noise  = stats.dist.normal(0, 0.02)

-- RNN COMPONENTS CONSTRUCTION SECTION
local g    = ann.graph() -- the RNN is a graph component
-- feed-forward components
local l1   = ann.components.hyperplane{ input=1, output=H }
local a1   = ann.components.actf.sin()
local l2   = ann.components.hyperplane{ input=H, output=1 }
local a2   = ann.components.actf.log_logistic()
-- gate components
local gates_actf     = "logistic"
local l1_input_gate  = ann.components.hyperplane{ input=H+1, output=1 }
local l1_output_gate = ann.components.hyperplane{ input=H+1, output=H }
local l1_forget_gate = ann.components.hyperplane{ input=H+1, output=H }
-- peephole component
local peephole = ann.graph.bind()
-- junction components
local l1_input  = ann.graph.cmul()
local l1_output = ann.graph.cmul()
local l1_forget = ann.graph.cmul()
-- recurrent junction component
local rec_add   = ann.graph.add()

-- RNN CONNECTIONS SECTION

-- feed-forward connections
g:connect('input', l1_input)( l1 )( rec_add )( a1 )( l1_output )( l2 )( a2 )( 'output' )
-- peephole recurrent connection
g:connect(a1, peephole )
g:connect('input', peephole)
-- gate connections
g:connect(peephole, l1_input_gate)( ann.components.actf[gates_actf]() )( l1_input )
g:connect(peephole, l1_output_gate)( ann.components.actf[gates_actf]() )( l1_output )
g:connect(peephole, l1_forget_gate)( ann.components.actf[gates_actf]() )( l1_forget )
-- recurrent connection
g:connect(l1_forget, rec_add)( l1_forget )

-- WEIGHTS INITIALIZATION SECTION, USING A TRAINER
trainable.supervised_trainer(g):build():
  randomize_weights{ inf=-0.1, sup=0.1, random=rnd1 }

iterator(ipairs(g.order)):map(function(k,v) return k,v,v:get_name() end):apply(print)

g:dot_graph("blah.dot")

g:set_bptt_truncation(BACKSTEP)

-- LOSS AND OPTIMIZER SECTION
local loss    = ann.loss.cross_entropy()
local opt     = ann.optimizer.adadelta()
local weights = g:copy_weights()
local keys    = iterator(table.keys(weights)):table() table.sort(keys)
for wname in iterator(table.keys(weights)):filter(function(k) return k:find("w.*") end) do
  opt:set_layerwise_option(wname, "weight_decay", WEIGHT_DECAY)
end

-- opt:set_option("max_iter", 1)

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- GENERATE INPUT/OUTPUT SAMPLE
local function build_input_output_sample(n)
  local input, sum = bern05:sample(rnd2, n), 0
  local output = matrix(1, 1, { input:sum() % 2 })
  return input, output
end

-- COMPUTE FORWARD
local function forward(g, input, during_training, it)
  g:reset(it)
  local input = input:clone():indexed_fill(1, input:select(2,1):lt(1), -1)
  if during_training then
    input:axpy(1.0, noise:sample(rnd2, input:size()))
  end
  local o_j
  local c = g:copy_components()
  for j=1,input:dim(1) do
    o_j = g:forward( input(j,':'), during_training )
    -- print("+++", table.concat(c.wop:get_output():toTable(), " "))
  end
  return o_j
end

-- CHECK GRADIENTS
local function check_gradients(x, y, weights, bptt_grads, EPSILON, func, ...)
  local keys       = iterator(table.keys(weights)):table() table.sort(keys)
  local diff_grads = {}
  for name,w in iterator(keys):map(function(k) return k,weights[k] end) do
    local bptt_grads = bptt_grads[name]:rewrap(bptt_grads[name]:size())
    local w = w:rewrap(w:size())
    for i=1,w:size() do
      local v = w:get(i)
      w:set(i, v + EPSILON)
      local y_hat = func(...)
      local l1 = loss:compute_loss(y_hat, y)
      w:set(i, v - EPSILON)
      local y_hat = func(...)
      local l2 = loss:compute_loss(y_hat, y)
      w:set(i, v)
      local g_bptt = bptt_grads:get(i)
      local g_diff = (l1-l2) / (2*EPSILON)
      if g_bptt ~= 0 or g_diff ~= 0 then
	local err = math.abs(g_diff - g_bptt) / math.max( math.abs(g_diff) + math.abs(g_bptt) )
        -- print(name, g_diff, g_bptt)
	assert( err < MAX_ERROR or g_bptt < EPSILON or g_diff < EPSILON, err )
      end
    end
  end
end

-- GRADIENT CHECKING LOOP
local function gradient_checking_loop()
  for sz=1,3 do
    for k=1,4 do
      local x,y = build_input_output_sample(sz)
      local y_hat = forward(g, x)
      g:backprop( loss:gradient(y_hat, y) )
      local bptt_grads = g:compute_gradients()
      check_gradients(x, y, weights, bptt_grads, EPSILON,
		      forward, g, x)
    end
  end
end

-- TRAINING LOOP
local function train(start, stop, max_seq_size)
    for i=start,stop do
      local sz = rnd2:randInt(1, max_seq_size)
      local input,output = build_input_output_sample(sz)
      local l,grads,y,o = opt:execute(function(x,it)
	  if x ~= weights then g:build{ weights=x } weights = x end
	  loss:reset()
	  local o_j = forward(g, input, true, it)
	  local y_j = output(1,':')
	  loss:accum_loss( loss:compute_loss( o_j, y_j  ) )
	  g:backprop( loss:gradient(o_j, y_j) )
	  local grads = g:compute_gradients()
	  return loss:get_accum_loss(),grads,y_j:get(1,1),o_j:get(1,1)
				      end, weights)
      printf("%5d  %.6f  ::  %.2f  %.2f  %5d :: %s\n", i, l,
	     y, (o<0) and math.exp(o) or o, sz,
	     table.concat(input:rewrap(input:size()):toTable(), " "))
    end
    return stop
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- TRAINING SECTION
if CHECK_GRADIENTS and BACKSTEP == math.huge then gradient_checking_loop() end
local last = 0
for s=2,MAX_SEQ_SIZE do
  last = train(last+1, last+MAX_EPOCHS, s)
end

-- EVALUATION SECTION
for i=1,1000 do
  local sz = rnd2:randInt(1, MAX_SEQ_SIZE*100)
  local x,y = build_input_output_sample(sz)
  local y_hat = forward(g, x):get(1,1)
  y = y:get(1,1)
  y_hat = (y_hat<0) and math.exp(y_hat) or y_hat
  printf("%.6f  ::  %.2f  %.2f  %d\n", math.abs(y_hat - y)^2,
	 y, y_hat, x:size())
  -- table.concat(x:rewrap(x:size()):toTable(), " "))
end

-- SHOW WEIGHTS SECTION
for wname,w in iterator(keys):map(function(k) return k,weights[k] end) do
  print("----------------------------")
  print(wname)
  print(w)
end
