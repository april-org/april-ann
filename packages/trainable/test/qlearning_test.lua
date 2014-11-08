-- MOUNTAIN CAR TEST (http://en.wikipedia.org/wiki/Mountain_Car)
local strategies  = trainable.qlearning_trainer.strategies

local num_runs = 10
local num_of_episodes = 101
local max_episode_length = 1000
local epsilon = 0.5 -- probability of random action, epsilon greedy
local epsilon_decay = 0.9999

local learning_rate = 0.1
local momentum = 0.4
local weight_decay = 1e-04
local L1_norm = 0.0
local max_norm_penalty = 4
local winf = -1
local wsup =  1
local bunch_size = 1024

local discount = 1.0 -- discount-rate parameter
local lambda = 0.9   -- trace-decay parameter

local weights_random = random(12384)
local exploration_random = random(2394)
local shuffle_random = random(23824)

------------------------------------------------------------------------------

-- two dimensions: velocity, position
local function start_state()
  return matrix(1,2,{0.0, -0.5})
end

local function stop_condition(state)
  return state:get(1,2) >= 0.6
end

-- three actions: left, neutral, right
local LEFT,NEUTRAL,RIGHT=1,2,3
local actions = { LEFT, NEUTRAL, RIGHT }
local function update(state, action)
  local action_values = { -1, 0, 1 }
  local next_state = matrix.as(state)
  local p = state:get(1,2)
  local v = state:get(1,1) + action_values[action] * 0.001 + math.cos(3 * p) * -0.0025
  local p = p + v
  next_state:set(1,1, math.clamp(v, -0.07, 0.07))
  next_state:set(1,2, math.clamp(p, -1.2, 0.6))
  local reward
  if stop_condition(next_state) then
    reward = 10
  else
    reward = 0
  end
  return next_state,reward
end

------------------------------------------------------------------------------

local best_length = max_episode_length/2
local best_net

for run=1,num_runs do
  ------------------------------------------------------------------------------
  local take_action = strategies.make_epsilon_decresing(actions,
                                                        epsilon,
                                                        epsilon_decay,
                                                        exploration_random)
  local thenet = ann.mlp.all_all.generate("2 inputs %d softmax" % { #actions })
  local sup_trainer = trainable.supervised_trainer(thenet,
                                                   ann.loss.mse(),
                                                   bunch_size,
                                                   ann.optimizer.rprop()):build()
  local qtrainer = trainable.qlearning_trainer{
    sup_trainer = sup_trainer,
    discount = discount,
    lambda = lambda,
    clampQ = function(v) return math.clamp(v,0.0,1.0) end,
  }
  qtrainer:randomize_weights{
    inf = winf,
    sup = wsup,
    random = weights_random,
  }
  if qtrainer:has_option("learning_rate") then
    qtrainer:set_option("learning_rate", learning_rate)
  end
  if qtrainer:has_option("momentum") then
    qtrainer:set_option("momentum", momentum)
  end
  if qtrainer:has_option("weight_decay") then
    qtrainer:set_option("weight_decay", weight_decay)
    qtrainer:set_option("L1_norm", L1_norm)
    qtrainer:set_option("max_norm_penalty", max_norm_penalty)
    for _,hyp in ipairs{ "weight_decay", "L1_norm", "max_norm_penalty" } do
      if qtrainer:has_option(hyp) then
        qtrainer:set_layerwise_option("b.*", hyp, 0)
      end
    end
  end
  local best
  for episode=1,num_of_episodes do
    collectgarbage("collect")
    -- RUN EPISODE
    qtrainer:reset() -- clear traces and other stuff
    local batch = qtrainer:get_batch_builder()
    local state,action = start_state(),NEUTRAL
    local output = qtrainer:calculate(state)
    local i = 0
    local acc_reward = 0
    while not stop_condition(state) and i < max_episode_length do
      local next_state,reward = update(state, action)
      acc_reward = acc_reward + reward
      if episode < num_of_episodes then
        batch:add(state, output, action, reward)
      end
      output = qtrainer:calculate(next_state) --qtrainer:one_step(action, state, reward)
      if episode < num_of_episodes then
        action = take_action(output)
      else
        _,action = output:max()
      end
      --print(i, table.concat(output:toTable(), " "), reward, action, table.concat(state:toTable(), " "))
      --
      i = i + 1
      --
      state = next_state
    end
    print(run, episode, i, acc_reward)
    if episode < num_of_episodes then
      -- BATCH TRAIN
      local in_ds,out_ds,mask_ds = batch:compute_dataset_pair()
      -- for ipat,pat in out_ds:patterns() do print(ipat, table.concat(pat, " "), table.concat(mask_ds:getPattern(ipat), " ")) end
      local train_func = trainable.train_wo_validation{
        max_epochs = (acc_reward > 0 and 100) or 10,
        min_epochs = (acc_reward > 0 and  40) or  2,
        percentage_stopping_criterion = 0.01,
      }
      while train_func:execute(function()
                                 local tr_error = sup_trainer:train_dataset{
                                   input_dataset  = in_ds,
                                   output_dataset = out_ds,
                                   mask_dataset = mask_ds,
                                   shuffle = shuffle_random,
                                 }
                                 return sup_trainer,tr_error
                               end) do
        -- print(train_func:get_state_string())
      end
    else -- if episode < num_of_episodes then ...
      if i < best_length then
        best_length = i
        best_net = sup_trainer:clone()
      end
    end -- if episode < num_of_episodes then ... else ...
  end
end

-- SHOW BEST RUN
if best_net then
  best_net:save("best_qlearning.net")
  local f = io.open("best_out.txt", "w")
  local state,action = start_state(),NEUTRAL
  local output = best_net:calculate(state)
  local i = 0
  while not stop_condition(state) do
    local next_state,reward = update(state, action)
    output = best_net:calculate(next_state)
    _,action = output:max()
    fprintf(f, "BEST %d %s %s %f %d\n",
            i,
            table.concat(state:toTable(), " "),
            table.concat(output:toTable(), " "),
            reward,
            action)
    --
    i = i + 1
    --
    state = next_state
  end
  f:close()
end
