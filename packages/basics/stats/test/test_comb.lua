local T = utest.test
local check = utest.check

function fact(n) return iterator(range(1,n)):reduce(math.mul(),1) end

T("CombinatorialsTest", function()
    for i=0,10 do
      for j=0,i do
        check.eq( stats.comb(i,j), fact(i) / (fact(j)*fact(i-j)),
                  string.format("C(%d,%d)", i, j) )
      end
    end
end)
