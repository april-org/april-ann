local trainable_qlearning_trainer_methods,
trainable_qlearning_trainer_class_metatable=class("trainable.qlearning_trainer")

-----------------------------
-- QLEARNING TRAINER CLASS --
-----------------------------

function trainable_qlearning_trainer_class_metatable:__call(t)
  local params = get_table_fields(
    {
      sup_trainer = { isa_match=trainable.supervised_trainer, mandatory=true },
      discount = { type_match="number", mandatory=true, default=0.6 },
      lambda = { type_match="number", mandatory=true, default=0.6 },
      gradients = { mandatory=false, default=matrix.dict() },
      traces = { mandatory=false, default=matrix.dict() },
      noise = { mandatory=false, default=ann.components.base() },
    }, t)
  local tr = params.sup_trainer
  local thenet  = tr:get_component()
  local weights = tr:get_weights_table()
  local optimizer = tr:get_optimizer()
  local obj = {
    tr = tr,
    thenet = thenet,
    weights = weights,
    optimizer = optimizer,
    gradients = params.gradients,
    traces = params.traces,
    discount = params.discount,
    lambda = params.lambda,
    noise = params.noise,
    nactions = thenet:get_output_size(),
  }
  return class_instance(obj, self)
end

-- PRIVATE METHOD
-- updates the weights given the previous state, the action, the current state
-- and the observed reward
function trainable_qlearning_trainer_train(self, prev_state, prev_action, state, reward)
  local noise = self.noise
  local thenet = self.thenet
  local optimizer = self.optimizer
  local gradients = self.gradients
  local traces = self.traces
  local nactions = self.nactions
  local discount = self.discount
  local lambda = self.lambda
  -- add random noise if given
  noise:reset(1)
  local prev_state = noise:forward(prev_state, true)
  noise:reset(0)
  local state = noise:forward(state, true)
  local error_grad = matrix.col_major(1, nactions):zeros()
  local loss,Qsp,Qs
  loss,gradients,Qsp,Qs,expected_Qsa =
    optimizer:execute(function(it)
                        assert(not it or it == 0)
                        thenet:reset(it)
                        local Qsp = thenet:forward(state):get_matrix()
                        local Qs  = thenet:forward(prev_state,true):get_matrix()
                        local Qsa = Qs:get(1, prev_action)
                        local expected_Qsa = math.min(1, math.max(0, reward + discount * Qsp:max()))
                        local diff = (Qsa - expected_Qsa)
                        local loss = 0.5 * diff * diff
                        error_grad:set(1, prev_action, ( Qsa - expected_Qsa ) )
                        thenet:backprop(error_grad)
                        gradients:zeros()
                        gradients = thenet:compute_gradients(gradients)
                        if traces:size() == 0 then
                          for name,g in pairs(gradients) do
                            traces[name] = matrix.as(g):zeros()
                          end
                        end
                        traces:scal(lambda)
                        traces:axpy(1.0, gradients)
                        return loss,traces,Qsp,Qs,expected_Qsa
                      end,
                      weights)
  self.gradients = gradients
  return loss,Qsp,Qs,expected_Qsa
end

-- takes the previos action, the current state (ANN input) and the reward,
-- updates the ANN weights and returns the current output Q(state,a)
function trainable_qlearning_trainer_methods:one_step(action, state, reward)
  local Qsp
  if self.prev_state then
    local loss,Qs,expected_Qsa
    loss,Qsp,Qs,expected_Qsa = trainer_train(self,
                                             self.prev_state,
                                             action,
                                             state,
                                             reward)
    self.Qprob = (self.Qprob or 0) + math.log(Qs:get(1,action))
    printf("%8.2f Q(s): %8.2f %8.2f %8.2f  E(Q(s)): %8.2f   ACTION: %d  REWARD: %6.2f  LOSS: %8.4f  MP: %.4f %.4f\n",
           -self.Qprob,
           Qs:get(1,1), Qs:get(1,2), Qs:get(1,3),
           expected_Qsa,
           action, reward, loss,
           self.tr:norm2("w."), self.tr:norm2("b."))
  else
    self.noise:reset(0)
    local state = self.noise:forward(state,true)
    Qsp = self.thenet:forward(state):get_matrix()
  end
  self.prev_state = state
  return Qsp
end

-- begins a new sequence of training
function trainable_qlearning_trainer_methods:reset()
  self.prev_state = nil
end

function trainable_qlearning_trainer_methods:set_option(...)
  self.tr:set_option(...)
end

function trainable_qlearning_trainer_methods:set_layerwise_option(...)
  self.tr:set_layerwise_option(...)
end

function trainable_qlearning_trainer_methods:get_option(...)
  self.tr:get_option(...)
end

function trainable_qlearning_trainer_methods:has_option(...)
  self.tr:has_option(...)
end

function trainable.qlearning_trainer.load(filename)
  return util.deserialize(filename)
end

function trainable_qlearning_trainer_methods:save(filename,format)
  util.serialize(self, filename, format)
end

function trainable_qlearning_trainer_methods:to_lua_string(format)
  return string.format("trainable.qlearning_trainer{%s}",
                       table.tostring({
                                        sup_trainer = self.tr,
                                        noise = self.noise,
                                        discount = self.discount,
                                        lambda = self.lambda,
                                        traces = self.traces,
                                        gradients = self.gradients,
                                      },
                                      format))
end
