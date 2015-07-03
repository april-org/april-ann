local T = utest.test
local check = utest.check
--
T("ROCTest", function()
    local roc_instance = metrics.roc()
    local output = matrix(100):uniformf(0,1,random(1234))
    local target = matrix(100):uniform(0,1,random(4567))
    roc_instance:add(output('1:50'), target('1:50'))
    roc_instance:add(output('51:100'), target('51:100'))
    check.number_eq( roc_instance:compute_area(), 0.5)
    check.number_eq( metrics.roc( matrix{0.2,0.3,0.4,0.5},
                                  matrix{0,1,0,1} ):compute_area(),
                     0.75 )
    check.eq( metrics.roc( matrix{0.2,0.3,0.4,0.5},
                           matrix{1,0,0,0} ):compute_area(),
              0.0 )
    check.number_eq( metrics.roc( matrix{0.2,0.3,0.4,0.5},
                                  matrix{0,0,1,0} ):compute_area(),
                     0.66 )
    check.number_eq( metrics.roc( matrix{0.2,0.3,0.4,0.5},
                                  matrix{0,0,0,1} ):compute_area(),
                     1.0 )
    --
    local t  = matrix{0,0,0,1,1,1}
    local d1 = matrix{0.2,0.3,0.4,0.5,0.2,0.9}
    local d2 = matrix{0.2,0.4,0.3,0.2,0.5,0.4}
    local roc1 = metrics.roc( d1, t )
    local roc2 = metrics.roc( d2, t )
    check.number_eq( roc1:compute_area(), 0.7222222354677)
    check.number_eq( roc2:compute_area(), 0.66666667660077)
    local h1 = metrics.roc.test(roc1,roc2,{method="bootstrap",seed=1234,verbose=true})
    local h2 = metrics.roc.test(roc1,roc2,{method="delong"})
    check.number_eq(h1:pvalue(), 0.89040226847281)
    check.number_eq(h2:pvalue(), 0.91152822827165)
    --
    local perm_result = stats.perm{
      samples = { d1, d2 },
      R = 1000, seed = 11234, verbose = true, k=1,
      statistic = function(d1,d2)
        return metrics.roc(d1,t):compute_area() - metrics.roc(d2,t):compute_area()
      end,
    }
    local AUC_diff = roc1:compute_area() - roc2:compute_area()
    check.number_eq(stats.perm.pvalue(perm_result, AUC_diff), 0.86*0.5)
end)
