base_dir = string.get_path(arg[0])
local rnd = random(567)
local errors = iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table()

local populations = stats.bootstrap_resampling{
  population_size = #errors,
  repetitions     = 1000,
  sampling_func   = function() return rnd:choose(errors) end,
  reducer         = stats.mean_var(),
}

table.sort(populations, function(a,b) return a[1] < b[1] end)

function check(a,b) return math.abs(a-b) < 1e-03 end

iterator(io.lines(base_dir.."test_bootstrap-output.log")):
map(string.tokenize):
enumerate():
apply(function(i,t)
        assert(check(tonumber(t[1]),populations[i][1]))
        assert(check(tonumber(t[2]),populations[i][2]))
      end)
