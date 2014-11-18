local gp = require "april_tools.gnuplot"()
local rnd = random(1234)
local x = stats.dist.normal(0,1):sample(rnd, matrix(2,1)):transpose()

-- A = inv([1, 1.98; 1.98, 4]);
local A = matrix(2,2,{ 50.251256, -24.874372,
                         -24.874372, 12.562814 })

function plot_samples(samples)
  local data = matrix(#samples,2)
  for i=1,#samples do
    data(i,':'):copy(samples[i]("x"))
  end
  local mu = data:sum(1)/data:dim(1)
  local data_centered = data:clone()
  for i=1,data_centered:dim(1) do
    data_centered(i,':'):axpy(-1.0, mu)
  end
  local stddev = (data^2):sum(1):scal(1/data:dim(1)):sqrt()
  local cov = (data_centered:transpose() * data_centered) / data:dim(1)
  print(mu)
  print(stddev)
  print(cov)
  gp:plot{ data=data, u='1:2', w='p' }
  io.read("*l")
  gp:close()
end

function correlated_normal(params)
  local grad = params.x * A
  local logp = -0.5 * grad:dot(params.x)
  return -logp, { x=grad }
end

----------------------------------------------------------------------------

local hmc = bayesian.optimizer.hmc()
hmc:set_option("seed", 4676)

hmc:start_burnin()
for i=1,5000 do hmc:execute(correlated_normal, {x=x}) end
hmc:finish_burnin()

for i=1,5000 do hmc:execute(correlated_normal, {x=x}) end
print(hmc:get_state_string())

plot_samples(hmc:get_samples())

