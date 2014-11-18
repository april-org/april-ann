local check = utest.check
local T = utest.test

T("MeansTest", function()
    local m = matrix(3,4):linear()
    check.number_eq(stats.amean(m), m:sum()/m:size())
    check.TRUE(stats.gmean(m) == 0)
    check.TRUE(stats.hmean(m) == 0)
    
    check.eq(stats.amean(m,1),
             matrix(1,4,{(0+4+8)/3, (1+5+9)/3, (2+6+10)/3, (3+7+11)/3}))
    check.eq(stats.gmean(m,1),
             matrix(1,4,{(0*4*8)^(1/3), (1*5*9)^(1/3), (2*6*10)^(1/3), (3*7*11)^(1/3)}))
    check.eq(stats.hmean(m,1),
             matrix(1,4,{0, 3/(1/1+1/5+1/9), 3/(1/2+1/6+1/10), 3/(1/3+1/7+1/11)}))
    
    local m = matrix(3,4):linear(1)
    check.number_eq(stats.amean(m), m:sum()/m:size())
    check.number_eq(stats.gmean(m),
                    iterator(range(1,12)):reduce(math.mul(),1)^(1/m:size()))
    check.number_eq(stats.hmean(m),
                    m:size() / iterator(range(1,12)):map(function(x) return 1/x end):reduce(math.add(),0))             
end)
