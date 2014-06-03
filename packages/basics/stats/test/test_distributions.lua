local check = utest.check
local M = matrix.col_major

-----------------------------------------------------------------------------
-- UNIFORM DISTRIBUTION

local d = stats.dist.uniform( M(3,{1,2,3}), M(3,{2,3,4}) )
local cdf_result = M(10,{ 
                       0.0592946, 0.21913, 0.472505, 0.0353412, 0.0440094,
                       0.124041, 0.0364003, 0.00314816, 0.161239, 0.0291983 })

check.eq( d:logcdf( d:sample( random(1234), M(10,3) ) ):exp(), cdf_result )

-----------------------------------------------------------------------------
-- NORMAL DISTRIBUTION

local d = stats.dist.normal( M(3):uniformf(0,1,random(1234)),
                             M(3,3):zeros():diag(2):set(1,2,1):set(2,1,1) )
check.eq(type(d), "stats.dist.normal.general")

local pdf_result = M(2, { 0.0175621, 0.00572678 })

check.eq( d:logpdf( M(2,3):uniformf(-1,2,random(9427)) ):exp(), pdf_result )

local N = 1000
local d = stats.dist.normal( M(3):fill(-10),
                             M(3,3):zeros():diag(2) )
local data = d:sample( random(4295), M(N,3) )
check(function()
        return data:sum(1):scal(1/N):equals( M(1,3):fill(-10), 0.1 )
      end)
local mv = stats.mean_var()
data:map(function(x) mv:add(x) end)
local mu,sigma = mv:compute()
check.number_eq(mu, -10)
check.number_eq(sigma, 2)

-----------------------------------------------------------------------------
-- DIAGONAL NORMAL DISTRIBUTION

local d2=d
local N = 1000
local d = stats.dist.normal( M(3):fill(-10), matrix.sparse.diag{2, 2, 2} )

check.eq(type(d), "stats.dist.normal.diagonal")
local data = d:sample( random(4295), M(N,3) )
check(function()
        return data:sum(1):scal(1/N):equals( M(1,3):fill(-10), 0.1 )
      end)
local mv = stats.mean_var()
data:map(function(x) mv:add(x) end)
local mu,sigma = mv:compute()
check.number_eq(mu, -10)
check.number_eq(sigma, 2)
local pdf_result = M(2,{-82.5465,-152.797})
check(function()
        return d:logpdf(M(2,3,{1,-5,3, -10,10,4})):equals(pdf_result,1e-02)
      end)

local d = stats.dist.normal(-30,4)
check(function()
        return d:sample(random(4243),10):equals( matrix.col_major(10,1,{
                                                                      -29.8856,
                                                                      -28.6116,
                                                                      -32.333,
                                                                      -27.8685,
                                                                      -30.8694,
                                                                      -31.9292,
                                                                      -27.3479,
                                                                      -29.7185,
                                                                      -29.3391,
                                                                      -30.6593,
                                                                       }),
                                                 1e-02)
      end)

-----------------------------------------------------------------------------
-- EXPONENTIAL DISTRIBUTION

local d = stats.dist.exponential(4)
local samples = M(10,1,{ 0.41319,
                         0.17446,
                         0.11866,
                         0.050273,
                         0.20654,
                         0.12271,
                         0.060404,
                         0.0649,
                         0.062123,
                         0.037511, })
check.eq( d:sample(random(1234),10), samples )
local pdf_result = matrix.col_major(10,{ 1.01885,
                                           -45.38,
                                           -1.25495,
                                           -15.5582,
                                           -11.3048,
                                           -26.279,
                                           -1.00439,
                                           -15.7705,
                                           -5.93778,
                                           -63.1531, })
local x = matrix.col_major(10,1,{ 0.091862,
                                  11.691586,
                                  0.660310,
                                  4.236131,
                                  3.172766,
                                  6.916327,
                                  0.597670,
                                  4.289197,
                                  1.831019,
                                  16.134842, })
check.eq( d:logpdf(x), pdf_result )

local cdf_result = matrix.col_major(10,{
                                        -1.17928,
                                        -4.89368e-21,
                                        -0.0739403,
                                        -4.37609e-08,
                                        -3.07852e-06,
                                        -9.66294e-13,
                                        -0.0960346,
                                        -3.53917e-08,
                                        -0.000659686,
                                        -9.35209e-29,
                                       })
check.eq( d:logcdf(x), cdf_result )

local x = x:rewrap(1,10)
local d = stats.dist.exponential(matrix.col_major(10):fill(4))

check.number_eq( d:logpdf(x):get(1), pdf_result:sum() )
check.number_eq( d:logcdf(x):get(1), cdf_result:sum() )
