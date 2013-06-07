class("stats.mean_var")

april_set_doc("stats.mean_var", {
		class       = "class",
		summary     = "Class to compute mean and variance",
		description ={
		  "This class is designed to compute mean and variance",
		  "by adding a sequence of data values (or tables)",
		}, })

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.__call", {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of a mean_var object",
		},
		params = {
		  "A number [optional]. If given, the assumed_mean approach",
		  "will be followed.",
		},
		outputs = { "A mean_var object" }, })

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

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.add", {
		class = "method", summary = "Adds one value",
		params = {
		  "A number",
		},
		outputs = { "The caller mean_var object (itself)" }, })

april_set_doc("stats.mean_var.add", {
		class = "method", summary = "Adds a sequence of values",
		params = {
		  "A Lua table (as array of numbers)",
		},
		outputs = { "The caller mean_var object (itself)" }, })

april_set_doc("stats.mean_var.add", {
		class = "method",
		summary = "Adds a value or values from a function call",
		params = {
		  "A Lua function",
		},
		outputs = { "The caller mean_var object (itself)" }, })

function stats.mean_var:add(v)
  if type(v) == "table" then
    for _,vp in ipairs(v) do return self:add(vp) end
  elseif type(v) == "function" then
    local vp = v()
    return self:add(vp)
  else
    local vi = v - self.assumed_mean
    self.accum_sum  = self.accum_sum + vi
    self.accum_sum2 = self.accum_sum2 + vi*vi
    self.N          = self.N + 1
  end
  return self
end

-----------------------------------------------------------------------------

april_set_doc("stats.mean_var.compute", {
		class = "method",
		summary = "Computes mean and variance of given values",
		outputs = {
		  "A number, the mean of the data",
		  "A number, the variance of the data",
		}, })

function stats.mean_var:compute()
  local mean,var
  local aux_mean = self.accum_sum / self.N
  mean = self.assumed_mean + aux_mean
  var  = (self.accum_sum2 - self.N * aux_mean * aux_mean) / (self.N - 1)
  return mean,var
end
