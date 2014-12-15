-- forces the use of CUDA
mathcore.set_use_cuda_default(util.is_cuda_available())
--

local check   = utest.check
local T       = utest.test

T("ANNGraphSourceTest", ann.graph.test)

T("ANNGraphComponentTest",
  function()
    
end)
