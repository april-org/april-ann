local check=utest.check
local T=utest.test

function do_test()
  T("GERTest", function()

      local t1 = { 3, 1, 9 }
      local t2 = { 9, 7, 10 }

      local t1_t2  = matrix(3,3,{ 3*9, 3*7, 3*10,
                                  1*9, 1*7, 1*10,
                                  9*9, 9*7, 9*10, })
      
      local t2_t1  = matrix(3,3,{ 3*9, 9*1, 9*9,
                                  3*7, 1*7, 9*7,
                                  3*10, 1*10, 9*10 })
      
      local X = matrix(t1)
      local Y = matrix(t2)
      
      check.eq(matrix(3,3):zeros():ger{ X=X, Y=Y, alpha=1 }, t1_t2)
      check.eq(matrix(3,3):zeros():ger{ X=Y, Y=X, alpha=1 }, t2_t1)
      
      check.errored(function() return check.eq(matrix(2,2):zeros():ger{ X=X, Y=Y, alpha=1 }, t1_t2) end)
  end)
end

do_test()
if util.is_cuda_available() then
  -- forces the use of CUDA
  mathcore.set_use_cuda_default(util.is_cuda_available())
  do_test()
end
