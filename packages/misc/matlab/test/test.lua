local check = utest.check
local T     = utest.test
--
local dir  = string.get_path(arg[0])
local test1 = matlab.read(dir.."test1.mat")
local test2 = matlab.read(dir.."test2.mat")
local test3 = matlab.read(dir.."test3.mat")
local test4 = matlab.read(dir.."test4.mat")

T("MatrixTest", function()
    local data=[[ 1.34187    -1.77726    -1.73478    -0.267141    0.67945    -0.864671    0.695942    1.12934    -1.25044    -0.710154  
-0.932328    0.59467     0.332692   -0.420799    0.1242      0.361931   -0.221126    0.779788   -0.00665479 -0.327634  
-0.254006   -2.86238     0.877438    0.200538    0.0397533  -0.127615   -0.429599    0.648093   -1.11046    -1.33459   
 1.87333    -0.572296    1.6083     -1.5067      0.707522    0.600558    0.589124    2.37575     0.0483031  -0.646724  
 1.3571     -1.62239    -1.25601    -0.480216    1.70552     0.00754286 -0.5679     -0.661177   -0.730054    1.2651    
 1.26294    -1.13459     0.799178   -0.0603598   0.967635   -1.4987     -0.419339   -1.61902    -0.597403    0.307181  
 0.885625   -0.303865    0.206848    0.669164    0.339318    2.30403     0.386978   -1.62213    -1.12447    -0.457475  
-0.130334    0.767747    2.3219     -1.03111    -0.389085    2.80222    -0.427865    0.995131   -0.948442   -0.0218299 
-0.999233    1.33908     0.968717   -0.68929     0.443295   -0.128687    0.590557    1.77715    -0.783717   -0.956102  
 0.564968    0.322677   -1.05338    -1.69783     1.7779      0.523945    0.810198    0.865949   -1.6916      0.827797]]
    local tgt = matrixDouble.read(aprilio.stream.input_lua_string(data),
                                  { tab=true })
    check.eq(tgt:convert_to("float"), test1.x:convert_to("float"))
end)

T("CellArrayTest", function()
    local data1="1 2 3\n4 5 6"
    local data2="7 8 9\n10 11 12"
    local tgt1=matrixDouble.read(aprilio.stream.input_lua_string(data1),
                                 { tab=true })
    local tgt2=matrixDouble.read(aprilio.stream.input_lua_string(data2),
                                 { tab=true })
    check.eq(test2.C:get(1,1):convert_to("float"), tgt1:convert_to("float"))
    check.eq(test2.C:get(1,2):convert_to("float"), tgt2:convert_to("float"))
end)

T("StructTest", function()
    check.eq(test3.X.y:get(1,1), 2)
    check.eq(test3.X.w:get(1,1), 1)
    check.eq(test3.X.z:get(1,1), 3)
end)

T("ComplexTest", function()
    local data = "1-2i 4-4i\n-1+2i -4+4i"
    local tgt = matrixComplex.read(aprilio.stream.input_lua_string(data),
                                   { tab=true })
    check.eq(test4.A, tgt)
end)
