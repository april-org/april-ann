-- MOUNTAIN CAR TEST (http://en.wikipedia.org/wiki/Mountain_Car)

local num_runs = 10
local num_of_episodes = 100
local max_episode_length = 10000
local epsilon = 0.1 -- probability of random action, epsilon greedy

local learning_rate = 0.0001
local momentum = 0.0
local weight_decay = 0.0
local L1_norm = 0.0
local max_norm_penalty = 4
local winf = -0.1
local wsup =  0.1

local discount = 0.9 -- discount-rate parameter
local lambda = 0.99   -- trace-decay parameter

local thenet = ann.mlp.all_all.generate("2 inputs 3 linear")
local sup_trainer = trainable.supervised_trainer(thenet):build()
local qtrainer = trainable.qlearning_trainer{
  sup_trainer = sup_trainer,
  discount = discount,
  lambda = lambda,
  clampQ = function(v) return math.clamp(v,0.0,1.0) end,
}

qtrainer:set_option("learning_rate", learning_rate)
qtrainer:set_option("momentum", momentum)
qtrainer:set_option("weight_decay", weight_decay)
qtrainer:set_option("L1_norm", L1_norm)
qtrainer:set_option("max_norm_penalty", max_norm_penalty)
for _,hyp in ipairs{ "weight_decay", "L1_norm", "max_norm_penalty" } do
  if qtrainer:has_option(hyp) then
    qtrainer:set_layerwise_option("b.*", hyp, 0)
  end
end

local weights_random = random(12384)
local exploration_random = random(2394)

------------------------------------------------------------------------------

-- two dimensions: velocity, position
local function start_state()
  return matrix.col_major(1,2,{0.0, -0.5})
end

local function stop_condition(state)
  return state:get(1,2) >= 0.6
end

-- three actions: left, neutral, right
local LEFT,NEUTRAL,RIGHT=1,2,3
local function update(state, action)
  local action_values = { -1, 0, 1 }
  local next_state = matrix.as(state)
  local p = state:get(1,2)
  local v = state:get(1,1) + action_values[action] * 0.001 + math.cos(3 * p) * -0.0025
  local p = p + v
  next_state:set(1,1, math.clamp(v, -0.07, 0.07))
  next_state:set(1,2, math.clamp(p, -1.2, 0.6))
  return next_state
end

local function take_action(output)
  local coin = exploration_random:rand()
  if coin < epsilon or output:max() - output:min() < 1e-02then
    return exploration_random:choose{LEFT,NEUTRAL,RIGHT}
  end
  local _,argmax = output:max()
  return argmax
end

for run=1,num_runs do
  qtrainer:randomize_weights{
    inf = winf,
    sup = wsup,
    random = weights_random,
  }
  local best
  for episode=1,num_of_episodes do
    collectgarbage("collect")
    -- RUN EPISODE
    qtrainer:reset() -- clear traces and other stuff
    local state,action,output = start_state(),NEUTRAL
    local i = 0
    local seq = { { action, state } }
    while not stop_condition(state) and i < max_episode_length do
      state  = update(state, action)
      local reward = math.clamp(state:get(1,2), -0.5, 0.5)
      output = qtrainer:one_step(action, state, reward)
      action = take_action(output)
      --print(i, table.concat(output:toTable(), " "), reward, action, table.concat(state:toTable(), " "))
      --
      i = i + 1
      table.insert(seq, { action, state })
    end
    if not best or #seq < #best then best = seq end
    print(run, episode, i)
  end
  --
  local state,action,output = start_state(),NEUTRAL
  local i = 0
  local seq = { { action, state } }
  while not stop_condition(state) and i < max_episode_length do
    state  = update(state, action)
    output = qtrainer.thenet:forward(state):get_matrix()
    _,action = output:max()
    -- print(i, table.concat(output:toTable(), " "), action, table.concat(state:toTable(), " "))
    --
    i = i + 1
    table.insert(seq, { action, state })
  end
  if not best or #seq < #best then best = seq end
  print(run, i)
  --
  local f = io.open("out-%04d.txt"%{run},"w")
  for i=1,#best do
    f:write(best[i][1])
    f:write(" ")
    f:flush()
    best[i][2]:toTabStream(f)
    f:flush()
  end
  f:close()
end
