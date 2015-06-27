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
    
    local roc1 = metrics.roc( matrix{0.2,0.3,0.4,0.5,0.2,0.9},
                              matrix{0,0,0,1,1,1} )
    local roc2 = metrics.roc( matrix{0.2,0.4,0.3,0.2,0.5,0.4},
                              matrix{0,0,0,1,1,1} )
    check.number_eq( roc1:compute_area(), 0.7222222354677)
    check.number_eq( roc2:compute_area(), 0.66666667660077)
    local h1 = metrics.roc.test(roc1,roc2,{method="bootstrap",seed=1234})
    local h2 = metrics.roc.test(roc1,roc2,{method="delong"})
    check.number_eq(h1:pvalue(), 0.89040226847281)
    check.number_eq(h2:pvalue(), 0.91152822827165)
end)
