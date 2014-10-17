local check = utest.check
local T = utest.test
local mstats = stats.mstats

T("MstatsTest", function()
    local m = matrix(3,4):linear()
    check.number_eq(mstats.amean(m), m:sum()/m:size())
    check.TRUE(mstats.gmean(m) == 0)
    check.TRUE(mstats.hmean(m) == 0)
    
    check.eq(mstats.amean(m,1),
             matrix(1,4,{(0+4+8)/3, (1+5+9)/3, (2+6+10)/3, (3+7+11)/3}))
    check.eq(mstats.gmean(m,1),
             matrix(1,4,{(0*4*8)^(1/3), (1*5*9)^(1/3), (2*6*10)^(1/3), (3*7*11)^(1/3)}))
    check.eq(mstats.hmean(m,1),
             matrix(1,4,{0, 3/(1/1+1/5+1/9), 3/(1/2+1/6+1/10), 3/(1/3+1/7+1/11)}))
    
    local m = matrix(3,4):linear(1)
    check.number_eq(mstats.amean(m), m:sum()/m:size())
    check.number_eq(mstats.gmean(m),
                    iterator(range(1,12)):reduce(math.mul(),1)^(1/m:size()))
    check.number_eq(mstats.hmean(m),
                    m:size() / iterator(range(1,12)):map(function(x) return 1/x end):reduce(math.add(),0))             
end)
