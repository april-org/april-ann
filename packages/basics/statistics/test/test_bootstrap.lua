local rnd = random(567)
local errors = iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table()

local populations = stats.bootstrap_resampling{
  population_size = #errors,
  repetitions     = 1000,
  sampling_func   = function() return rnd:choose(errors) end,
  reducer         = stats.mean_var(),
}

table.sort(populations, function(a,b) return a[1] < b[1] end)

iterator(ipairs(populations)):select(2):field(1,2):apply(print)
