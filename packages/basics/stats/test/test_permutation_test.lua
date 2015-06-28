local check = utest.check
local T = utest.test
local base_dir = string.get_path(arg[0])
local rnd = random(567)
local errors1 = matrix(iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table())
local errors2 = matrix(iterator(range(1,1000)):map(function()return rnd:randNorm(0.01,1.0)end):table())

T("PermutationTest",
  function()
    local perm_result = stats.perm{
      samples={ errors1, errors2 },
      R=1000, seed=1234, verbose=true, k=1,
      statistic = function(e1, e2)
        local m1 = stats.amean(e1)
        local m2 = stats.amean(e2)
        return m1 - m2
      end
    }
    local perm_result = perm_result:index(1, perm_result:select(2,1):order())
    local mean_diff = stats.amean(errors1) - stats.amean(errors2)
    check.number_eq(mean_diff, -0.041067314147949)
    check.number_eq(stats.perm.pvalue(perm_result, mean_diff), 0.368)
end)
