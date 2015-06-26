local check = utest.check
local T = utest.test
local base_dir = string.get_path(arg[0])
local rnd = random(567)
local errors = matrix(iterator(range(1,1000)):map(function()return rnd:randNorm(0.0,1.0)end):table())

T("BootstrapTest",
  function()
    local boot_result = stats.boot{
      size=errors:size(), R=1000, seed=1234, verbose=true, k=2,
      statistic = function(sample)
        local s = errors:index(1, sample)
        local var,mean = stats.var(s)
        return mean,var
      end
    }
    local boot_result = boot_result:index(1, boot_result:select(2,1):order())
    iterator(io.lines(base_dir.."test_bootstrap-output.log")):
      map(string.tokenize):
      enumerate():
      apply(function(i,t)
          check.number_eq(tonumber(t[1]), boot_result[i][1])
          check.number_eq(tonumber(t[2]), boot_result[i][2])
      end)
    --
    local boot_result = stats.boot{
      size=errors:size(), R=1000, random=random(1234), ncores=2, verbose=true,
      k=2,
      statistic = function(sample)
        local s = errors:index(1, sample)
        local var,mean = stats.var(s)
        return mean,var
      end
    }
    local boot_result = boot_result:index(1, boot_result:select(2,1):order())
    iterator(io.lines(base_dir.."test_bootstrap-output.log")):
      map(string.tokenize):
      enumerate():
      apply(function(i,t)
          check.number_eq(tonumber(t[1]), boot_result[i][1])
          check.number_eq(tonumber(t[2]), boot_result[i][2])
      end)
    local a,b = stats.boot.ci(boot_result, 0.95)
    local m,p0,pn = stats.boot.percentile(boot_result, { 0.5, 0.0, 1.0 })
    check.number_eq(a,  -0.073266)
    check.number_eq(b,   0.051443)
    check.number_eq(m,  -0.012237)
    check.number_eq(p0, -0.117945)
    check.number_eq(pn,  0.092709)
    -- p-value tests
    check.number_eq(stats.boot.rprob(boot_result, b), 0.025, 0.05)
    check.number_eq(1.0 - stats.boot.rprob(boot_result, a), 0.025, 0.05)
    check.number_eq(stats.boot.rprob(boot_result, m, 1), 0.500)
end)
