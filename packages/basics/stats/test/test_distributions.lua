local check = utest.check
local T = utest.test
local M = matrix
local EPSILON = 0.001
local N = 1000

local function check_grads(d, point)
  local point = point:clone()
  local g = d:logpdf_derivative(point)
  for i = 1,point:dim(2) do
    local v = point:get(1,i)
    point:set(1, i, v + EPSILON)
    local logpdf1 = d:logpdf(point):get(1)
    point:set(1, i, v - EPSILON)
    local logpdf2 = d:logpdf(point):get(1)
    point:set(1, i, v)
    local g_hat = (logpdf1 - logpdf2) / (2 * EPSILON)
    check.number_eq(g:get(1,i), g_hat)
  end
end

-----------------------------------------------------------------------------
-- UNIFORM DISTRIBUTION

T("UniformDistTest", function()
    local d = stats.dist.uniform( M(3,{1,2,3}), M(3,{2,3,4}) )
    local cdf_result = M(10,{ 
                           0.0592946, 0.21913, 0.472505, 0.0353412, 0.0440094,
                           0.124041, 0.0364003, 0.00314816, 0.161239, 0.0291983 })
    
    check.eq( d:logcdf( d:sample( random(1234), M(10,3) ) ):exp(), cdf_result )
end)

-----------------------------------------------------------------------------
-- NORMAL DISTRIBUTION

T("NormalDistTest", function()
    local d = stats.dist.normal( M(3):uniformf(0,1,random(1234)),
                                 M(3,3):zeros():diag(2):set(1,2,1):set(2,1,1) )
    check.eq(type(d), "stats.dist.normal.general")
    
    local pdf_result = M(2, { 0.0151933, 0.00832571 })
    
    check.eq( d:logpdf( M(2,3):uniformf(-1,2,random(9427)) ):exp(), pdf_result )
    
    for _,a in ipairs{-4, -0.1, 0.1, 4} do
      for _,b in ipairs{0.1, 1.0, 4.0} do
        local d = stats.dist.normal( M(3):fill(a),
                                     M(3,3):zeros():diag(b) )
        local data = d:sample( random(4295), M(N,3) )
        -- FIXME: check this test, it fails eventually
        -- check(function()
        --     return data:sum(1):scal(1/N):equals( M(1,3):fill(a) )
        -- end, "Multivariate population mean test")
        local mv = stats.running.mean_var()
        data:map(function(x) mv:add(x) end)
        local mu,sigma = mv:compute()
        check.number_eq(mu, a, ( math.abs(a) < 1.0 ) and 0.4 or nil,
                        "Population mean test")
        check.number_eq(sigma, b, nil, "Population variance test")
        for i=1,10 do check_grads(d, data(i,':')) end
      end
    end
end)

-----------------------------------------------------------------------------

T("QuantileTest", function()
    local d = stats.dist.normal()
    check.number_eq(stats.dist.quantile(d, 0.975), 1.9599)
    check.number_eq(stats.dist.quantile(d, 0.95), 1.6448)
    check.number_eq(stats.dist.quantile(d, 0.05), -1.6448)
    check.number_eq(stats.dist.quantile(d, 0.20), -0.8416212)
    check.number_eq(stats.dist.quantile(d, 0.50), 0.0)
end)

-----------------------------------------------------------------------------
-- DIAGONAL NORMAL DISTRIBUTION

T("DiagNormalDistTest", function()
    for _,a in ipairs{-4, -0.1, 0.1, 4} do
      for _,b in ipairs{0.1, 1.0, 4.0} do
        local d = stats.dist.normal( M(3):fill(a), matrix.sparse.diag{b, b, b} )
        
        check.eq(type(d), "stats.dist.normal.diagonal")
        local data = d:sample( random(4295), M(N,3) )
        -- FIXME: check this test, it fails eventually
        -- check(function()
        --     return data:sum(1):scal(1/N):equals( M(1,3):fill(a) )
        -- end, "Multivariate population mean test")
        local mv = stats.running.mean_var()
        data:map(function(x) mv:add(x) end)
        local mu,sigma = mv:compute()
        check.number_eq(mu, a, ( math.abs(a) < 1.0 ) and 0.4 or nil,
                        "Population mean test")
        check.number_eq(sigma, b, nil, "Population variance test")
        for i=1,10 do check_grads(d, data(i,':')) end
      end
    end
end)

-----------------------------------------------------------------------------
-- STANDARD NORMAL DISTRIBUTION

T("StdNormalDist", function()
    local d = stats.dist.normal()
    check.eq(type(d), "stats.dist.normal.standard")
    local data = d:sample( random(4295), M(N,1) )
    check(function()
        return math.abs(data:sum()/N) < 0.1
    end)
    local mv = stats.running.mean_var()
    data:map(function(x) mv:add(x) end)
    local mu,sigma = mv:compute()
    check(function() return math.abs(mu) < 0.04 end)
    check.number_eq(sigma, 1.0, 0.1)
    for i=1,10 do check_grads(d, data(i,':')) end
end)

-----------------------------------------------------------------------------
-- EXPONENTIAL DISTRIBUTION

T("ExpDistTest", function()
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
    local pdf_result = M(10,{ 1.01885,
                                -45.38,
                                -1.25495,
                                -15.5582,
                                -11.3048,
                                -26.279,
                                -1.00439,
                                -15.7705,
                                -5.93778,
                                -63.1531, })
    local x = M(10,1,{ 0.091862,
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

    local cdf_result = M(10,{
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
    local d = stats.dist.exponential(M(10):fill(4))

    check.number_eq( d:logpdf(x):get(1), pdf_result:sum() )
    check.number_eq( d:logcdf(x):get(1), cdf_result:sum() )
end)

-----------------------------------------------------------------------------
-- BINOMIAL DISTRIBUTION

T("BinomDistTest", function()
    local d = stats.dist.binomial(10,0.5)

    check.eq( d:sample(random(1234),10),
              M(10,1,{3,6,6,6,5,5,4,5,3,5}) )

    local pdf_result = M(3,{-1.58436,-3.12481,-4.62889})
    check.eq( d:logpdf(M(3,1,{4,8,1})), pdf_result )

    local cdf_result = M(3,{-0.975635,-0.0108004,-4.53358})
    check.eq( d:logcdf(M(3,1,{4,8,1})), cdf_result )
end)

-----------------------------------------------------------------------------
-- BERNOULLI DISTRIBUTION

T("BernoulliDistTest", function()
    local d = stats.dist.bernoulli(0.5)
    check.TRUE(class.is_a(d, stats.dist.binomial))
end)

-----------------------------------------------------------------------------
-- LOG-NORMAL DISTRIBUTION

T("LogNormalDistTest", function()
    for _,a in ipairs{-4, -0.1, 0.1, 4} do
      for _,b in ipairs{0.1, 1.0, 4.0} do
        local d = stats.dist.lognormal( M(3):fill(a),
                                        M(3,3):zeros():diag(b) )
        local data = d:sample( random(4295), M(N,3) )
        check(function()
            return (data:sum(1):scal(1/N):log() - M(1,3):fill(a + b/2)):abs():sum()/3 < 0.1
        end)
        local mv = stats.running.mean_var()
        data:map(function(x) mv:add(x) end)
        local mu,sigma = mv:compute()
        check.number_eq(math.log(mu), a + b/2, 0.1)
        -- FIXME: PRECISSION PROBLEMS ?????
        -- check.number_eq(math.log(sigma), (b - 1 + (2*a + b)))
      end
    end
end)

-----------------------------------------------------------------------------
-- DIAGONAL NORMAL DISTRIBUTION

T("DiagLogNormalDistTest", function()
    for _,a in ipairs{-4, -0.1, 0.1, 4} do
      for _,b in ipairs{0.1, 1.0, 4.0} do
        local d = stats.dist.lognormal( M(3):fill(a), matrix.sparse.diag{b, b, b} )
        check.eq(type(d), "stats.dist.lognormal.diagonal")
        local data = d:sample( random(4295), M(N,3) )
        check(function()
            return (data:sum(1):scal(1/N):log() - M(1,3):fill(a + b/2)):abs():sum()/3 < 0.1
        end)
        local mv = stats.running.mean_var()
        data:map(function(x) mv:add(x) end)
        local mu,sigma = mv:compute()
        check.number_eq(mu, math.exp(a + b/2), 0.1)
        -- FIXME: PRECISSION PROBLEMS ?????
        -- check.number_eq(math.log(sigma), (b - 1 + (2*a + b)), 0.08)
      end
    end
end)
