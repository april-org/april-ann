local T = utest.test
local check = utest.check

T("KMeansMatrixRefineTest", function()
    local filename = arg[1] or string.get_path(arg[0]) .. 'data.txt'
    local data = matrix.fromFilename(filename)
    local K = 2
    local res,C = clustering.kmeans.matrix({ 
        data = data,
        K = K,
        random = random(1234)
    })
    check.number_eq(res, 0.053191109336913, nil, "result check")
    check.eq(C, matrix(2,16,
                       {
                           -1.28483,    -0.798679,    0.282941,   -0.365985,   -0.942726,    0.103687,   -1.52005,     0.176974,   -0.0922824,  -0.481356,    1.38082,    -1.02229,     0.0243497,   0.133867,   -0.588762,   -0.853817,  
                           -1.51381,    -1.09521,     0.276019,   -0.352453,   -1.08768,     0.167996,   -1.64307,     0.227699,   -0.0660796,  -0.596381,    1.70157,    -1.14844,    -0.0181014,   0.187125,   -0.618449,   -0.946881,  
                         
                      }),
             "clusters check")
end)
