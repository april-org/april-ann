local CHECK_GRADIENTS = true
local BACKSTEP        = math.huge
local MAX_ERROR       = 0.04
local EPSILON         = 0.01
local MAX_SEQ_SIZE    = 10
local SEQ_STEP        = 1
local MAX_EPOCHS      = 1000 -- max epochs for sequence size = 2,MAX_SEQ_SIZE
local WEIGHT_DECAY    = 0.00001
local H               = 2 -- number of neurons in hidden layer

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local rnd1   = random(7576)
local rnd2   = random(1234)
local rnd3   = random(5723)
local rnd4   = random(4825)
local rnd5   = random(8257)
local bern05 = stats.dist.bernoulli(0.5)
local noise  = stats.dist.normal(0, 0.01)

-- RNN COMPONENTS CONSTRUCTION SECTION
local g    = ann.graph('parity') -- the RNN is a graph component
-- by default all gates and peepholes are true
local lstm = ann.graph.blocks.lstm({ input=1, output=H,
                                     actf="softsign", name="LSTM" })
g:connect( 'input', lstm,
           ann.components.hyperplane{ input=H, output=1, name="l2" },
           ann.components.actf.log_logistic{ name="a2" },
           'output' )

-- WEIGHTS INITIALIZATION SECTION, USING A TRAINER
local trainer = trainable.supervised_trainer(g):build{ input=1, output=1 }
trainer:randomize_weights{ inf=-0.1, sup=0.1, random=rnd1 }

if lstm:get_is_built() then lstm:dot_graph("jaja.dot") end

--[[
1       instance of ann.graph.bind      c24
2       instance 0x21d9060 of ann.components.hyperplane c16
3       instance 0x21dd190 of ann.components.actf.logistic      c28
4       instance of ann.graph.cmul      c21
5       instance of ann.graph.bind      c22
6       instance 0x21cdab0 of ann.components.hyperplane c10
7       instance 0x21dda70 of ann.components.actf.logistic      c26
8       instance 0x21cb660 of ann.components.hyperplane c2
9       instance of ann.graph.cmul      c19
10      instance of ann.graph.add       c25
11      instance of ann.graph.bind      c23
12      instance 0x21d8dc0 of ann.components.hyperplane c13
13      instance 0x21de790 of ann.components.actf.logistic      c27
14      instance 0x21cb850 of ann.components.actf.tanh  c5
15      instance of ann.graph.cmul      c20
16      instance 0x21cd750 of ann.components.hyperplane c6
17      instance 0x21cd900 of ann.components.actf.log_logistic  c9
]]

g:show_nodes()

g:dot_graph("blah.dot")

if g:get_is_recurrent() then g:set_bptt_truncation(BACKSTEP) end

-- LOSS AND OPTIMIZER SECTION
local loss    = ann.loss.cross_entropy()
local opt     = ann.optimizer.adadelta()
local weights = g:copy_weights()
local keys    = iterator(table.keys(weights)):table() table.sort(keys)

local function set_weight_decay(opt)
  for wname in iterator(table.keys(weights)):filter(function(k) return k:find(".*") end) do
    opt:set_layerwise_option(wname, "weight_decay", WEIGHT_DECAY)
  end
  return opt
end

set_weight_decay(opt)

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- GENERATE INPUT/OUTPUT SAMPLE
local function build_input_output_sample(n, rnd)
  local input, sum = bern05:sample(rnd, n), 0
  local output = matrix(1, 1, { input:sum() % 2 })
  return input, output
end

-- COMPUTE FORWARD
local function forward(g, input, during_training, it)
  g:reset(it)
  local input = input:clone():indexed_fill(1, input:select(2,1):lt(1), -1)
  if during_training then
    input:axpy(1.0, noise:sample(rnd4, input:size()))
  end
  local o_j
  local c = g:copy_components()
  for j=1,input:dim(1) do
    o_j = g:forward( input(j,':'), during_training )
    -- print("iii", table.concat(c.c19:get_output():toTable(), " "))
    -- print("@@@", table.concat(c.c21:get_output():toTable(), " "))
    -- print("---", iterator(c.c25:get_input():iterate()):map(function(k,v) return "|||",table.concat(v:toTable(), " ") end):concat(" "," "))
    -- print("+++", table.concat(c.c25:get_output():toTable(), " "))
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
      -- print(name, g_diff, g_bptt)
      if g_bptt ~= 0 or g_diff ~= 0 then
	local err = math.abs(g_diff - g_bptt) / math.max( math.abs(g_diff) + math.abs(g_bptt) )
        assert( err < MAX_ERROR or g_bptt < EPSILON or g_diff < EPSILON, err )
      end
    end
  end
end

-- GRADIENT CHECKING LOOP
local function gradient_checking_loop()
  for sz=1,3 do
    for k=1,4 do
      -- print("SIZE ", sz, "K ", k)
      local x,y = build_input_output_sample(sz, rnd5)
      local y_hat = forward(g, x)
      g:backprop( loss:gradient(y_hat, y) )
      local bptt_grads = g:compute_gradients()
      local bptt_data = g:get_bptt_state()
      check_gradients(x, y, weights, bptt_grads, EPSILON,
		      forward, g, x)
    end
  end
end

-- TRAINING LOOP
local function train(start, stop, max_seq_size, opt)
    for i=start,stop do
      local sz = rnd2:randInt(1, max_seq_size)
      local input,output = build_input_output_sample(sz, rnd2)
      local l,grads,y,o = opt:execute(function(x,it)
	  if x ~= weights then g:build{ weights=x } weights = x end
	  loss:reset()
	  local o_j = forward(g, input, true, it)
	  local y_j = output(1,':')
	  loss:accum_loss( loss:compute_loss( o_j, y_j  ) )
	  g:backprop( loss:gradient(o_j, y_j) )
	  local grads = g:compute_gradients()
          local m = matrix.dict.iterator(grads):select(2):
            map(matrix.op.abs):map(matrix.."max"):select(1):reduce(math.max,0)
	  return loss:get_accum_loss(),grads,y_j:get(1,1),o_j:get(1,1)
				      end, weights)
      printf("%5d  %.6f  ::  %.2f  %.2f  %5d :: %.6f %.6f :: %s\n", i, l,
	     y, (o<0) and math.exp(o) or o, sz,
             trainer:norm2("w.*"), trainer:norm2("b.*"),
	     input:size() < 10 and table.concat(input:rewrap(input:size()):toTable(), " ") or " ")
    end
    return stop
end

local function save_activations(g, sz, filename)
  local input       = matrix(1, sz)
  local input_gate  = matrix(H, sz)
  local output_gate = matrix(H, sz)
  local forget_gate = matrix(H, sz)
  local hidden      = matrix(H, sz)
  local output      = matrix(1, sz)
  local input_name  = g:get_input_name()
  local output_name = g:get_output_name()
  for t=1,sz do
    local aux = {':',t}
    local state = g:get_bptt_state(t)
    input[aux] = state[input_name].input:t()
    output[aux] = state[output_name].output:t()
    input_gate[aux] = state["LSTM::i::gate"].output:t()
    output_gate[aux] = state["LSTM::o::gate"].output:t()
    forget_gate[aux] = state["LSTM::f::gate"].output:t()
    hidden[aux] = state["LSTM::memory"].output:t()
  end
  local r = matrix.join(1, input, input_gate, forget_gate, hidden,
                        output_gate, matrix.op.exp(output))
  ImageIO.write(Image(r), filename)
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

-- TRAINING SECTION
if CHECK_GRADIENTS and BACKSTEP == math.huge then gradient_checking_loop() end
local last = 0
for s=2,MAX_SEQ_SIZE,SEQ_STEP do
  last = train(last+1, last+MAX_EPOCHS, s, opt)
end

-- EVALUATION SECTION
local loss = ann.loss.zero_one()
for i=1,1000 do
  local sz = rnd3:randInt(1, MAX_SEQ_SIZE*100)
  local x,y = build_input_output_sample(sz, rnd3)
  local y_hat = forward(g, x)
  if i==1 then save_activations(g, sz, "activations_lstm.png") end
  loss:accum_loss( loss:compute_loss(matrix.op.exp(y_hat), y) )
  y     = y:get(1,1)
  y_hat = y_hat:get(1,1)
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

print("----------------------------")
print(loss:get_accum_loss())
