class("stats.mean_var")

function stats.mean_var:__call(assumed_mean)
  local obj = {
    assumed_mean = assumed_mean or 0,
    accum_sum    = 0,
    accum_sum2   = 0,
    N            = 0,
  }
  class_instance(obj, self, true)
  return obj
end

function stats.mean_var:add(v)
  local vi = v - self.assumed_mean
  self.accum_sum  = self.accum_sum + vi
  self.accum_sum2 = self.accum_sum2 + vi*vi
  self.N          = self.N + 1
end

function stats.mean_var:compute()
  local mean,var
  local aux_mean = self.accum_sum / self.N
  mean = self.assumed_mean + aux_mean
  var  = (self.accum_sum2 - self.N * aux_mean * aux_mean) / (self.N - 1)
  return mean,var
end
