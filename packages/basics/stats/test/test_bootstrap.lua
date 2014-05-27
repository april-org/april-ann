local check = utest.check
local base_dir = string.get_path(arg[0])
local rnd = random(567)
local errors = iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table()

local boot_result = stats.boot{
  data=errors, R=1000, seed=1234,
  statistic = function(it)
    local mv = stats.mean_var()
    for k,v in it do mv:add(v) end
    return { mv:compute() }
  end
}
table.sort(boot_result, function(a,b) return a[1]<b[1] end)
iterator(io.lines(base_dir.."test_bootstrap-output.log")):
map(string.tokenize):
enumerate():
apply(function(i,t)
        check.lt(math.abs(tonumber(t[1]) - boot_result[i][1]), 1e-03)
        check.lt(math.abs(tonumber(t[2]) - boot_result[i][2]), 1e-03)
      end)
