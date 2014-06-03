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
check.lt( math.abs(mu + 10), 0.1 )
check.lt( math.abs(sigma - 2), 0.1 )

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
check.lt( math.abs(mu + 10), 0.1 )
check.lt( math.abs(sigma - 2), 0.1 )
local pdf_result = M(2,{-82.5465,-152.797})
check(function()
        return d:logpdf(M(2,3,{1,-5,3, -10,10,4})):equals(pdf_result,1e-02)
      end)
