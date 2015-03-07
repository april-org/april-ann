local T = utest.test
local check = utest.check

function log_comb(n, k)
  return stats.lgamma(n+1) - stats.lgamma(k+1) - stats.lgamma(n-k+1)
end

function comb(n, k)
  return math.round(math.exp(log_comb(n, k)))
end

T("CombinatorialsTest", function()
    for i=0,30 do
      for j=0,i do
        check.eq( stats.comb(i,j),
                  comb(i,j),
                  string.format("C(%d,%d)", i, j) )
      end
    end
end)

T("LogCombinatorialsTest", function()
    for i=0,100 do
      for j=0,i do
        check.number_eq( stats.log_comb(i,j),
                         log_comb(i,j),
                         0.01, -- 1% relative error
                         string.format("C(%d,%d)", i, j) )
      end
    end
end)
