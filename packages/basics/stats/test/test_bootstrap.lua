local check = utest.check
local base_dir = string.get_path(arg[0])
local rnd = random(567)
local errors = iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table()

local populations = stats.bootstrap_resampling{
  population      = errors,
  repetitions     = 1000,
  initial         = function() return stats.mean_var() end,
  reducer         = function(acc,v) return acc:add(v) end,
  postprocess     = function(acc) return acc:compute() end,
  seed            = 1234,
}

table.sort(populations, function(a,b) return a[1] < b[1] end)

iterator(io.lines(base_dir.."test_bootstrap-output.log")):
map(string.tokenize):
enumerate():
apply(function(i,t)
        check.eq(tonumber(t[1]),populations[i][1])
        check.eq(tonumber(t[2]),populations[i][2])
      end)
