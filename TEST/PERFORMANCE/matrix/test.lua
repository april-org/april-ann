local arg = arg or {}
local use_cuda = (table.remove(arg,1)=="true")
if util.is_cuda_available() and use_cuda then
  mathcore.set_use_cuda_default(true)
  printf("USING CUDA\n\n")
end
--
local tests_list = table.invert(arg)
--
local clock = util.stopwatch()
local rnd   = random(1234)

local T = function(id, func)
  local id=id:gsub(" ","_")
  if #arg == 0 or tests_list[id] then
    print(id)
    pcall(func)
    print()
  end
end

local measure_process_time = function(func,...)
  collectgarbage("collect")
  local it=0
  clock:reset()
  clock:go()
  repeat
    if not pcall(func,...) then collectgarbage("collect") end
    it=it+1
  until clock:read() > 0.1
  clock:stop()
  local a,b = clock:read()
  return a/it,b/it
end

local root = function (N,r,M)
  if r == 1 then return N,M end
  local v = math.floor(N^(1/r))
  local tbl = iterator(range(1,r)):map(function() return v end):table()
  if M then table.insert(tbl,M) end
  return table.unpack(tbl)
end

T("SUM 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:sum() end))
    end
end)

T("SUM 2D TEST", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2)):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:sum() end))
    end
end)

T("SUM 3D TEST", function()
    for i=1,8 do
      local N = 9^i
      local m = matrix(root(N,3)):uniformf(-1,1,rnd)
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"3D",
	     measure_process_time(function() m:sum() end))
    end
end)

T("SUM 4D TEST", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,4)):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"4D",
	     measure_process_time(function() m:sum() end))
    end
end)

-----------------------------------------------------------------------------

T("SUM 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:sum() end))
    end
end)

T("SUM 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:sum() end))
    end
end)

T("SUM 3D TEST SUBMATRIX", function()
    for i=1,8 do
      local N = 9^i
      local m = matrix(root(N,3,2)):uniformf(-1,1,rnd):select(4,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"3D",
	     measure_process_time(function() m:sum() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:clone() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+SCAL 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:clone():scal(10) end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+SCAL 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone():scal(10) end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+SCAL 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:clone():scal(10) end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+EXP 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone():exp() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+EXP 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone():exp() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+EXP 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:clone():exp() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+LOG 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone():log() end))
    end
end)

-----------------------------------------------------------------------------

T("CLONE+LOG 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:clone():log() end))
    end
end)


-----------------------------------------------------------------------------

T("CLONE+LOG 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:clone():log() end))
    end
end)


-----------------------------------------------------------------------------

T("FILL 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:fill(10) end))
    end
end)

-----------------------------------------------------------------------------

T("FILL 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:fill(10) end))
    end
end)

-----------------------------------------------------------------------------

T("FILL 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:fill(10) end))
    end
end)

-----------------------------------------------------------------------------

T("GEMM", function()
    for i=4,11 do
      local N = 2^i
      local a = matrix(N,N):uniformf(-1,1,rnd) a:update()
      local b = matrix(N,N):uniformf(-1,1,rnd) b:update()
      local c = matrix(N,N):uniformf(-1,1,rnd) c:update()
      printf("\tsize=%10d      %20.9f  %20.9f\n",c:size(),
	     measure_process_time(function()
		 c:gemm{ A=a, B=b, alpha=1.0, beta=1.0 }
      end))
    end
end)

-----------------------------------------------------------------------------

T("GEMV", function()
    for i=4,11 do
      local N = 2^i
      local a = matrix(N,N):uniformf(-1,1,rnd) a:update()
      local x = matrix(N):uniformf(-1,1,rnd) x:update()
      local y = matrix(N):uniformf(-1,1,rnd) y:update()
      printf("\tsize=%10d      %20.9f  %20.9f\n",a:size(),
	     measure_process_time(function()
		 y:gemv{ A=a, X=x, alpha=1.0, beta=1.0 }
      end))
    end
end)

-----------------------------------------------------------------------------

T("NORM2 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N):uniformf(-1,1,rnd) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:norm2() end))
    end
end)

-----------------------------------------------------------------------------

T("NORM2 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local m = matrix(N,2):uniformf(-1,1,rnd):select(2,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"1D",
	     measure_process_time(function() m:norm2() end))
    end
end)

-----------------------------------------------------------------------------

T("NORM2 2D TEST SUBMATRIX", function()
    for i=4,13 do
      local N = 4^i
      local m = matrix(root(N,2,2)):uniformf(-1,1,rnd):select(3,1) m:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",m:size(),"2D",
	     measure_process_time(function() m:norm2() end))
    end
end)

-----------------------------------------------------------------------------

T("DOT 1D TEST", function()
    for i=1,7 do
      local N = 10^i
      local a = matrix(N):uniformf(-1,1,rnd) a:update()
      local b = matrix(N):uniformf(-1,1,rnd) b:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",a:size(),"1D",
	     measure_process_time(function() a:dot(b) end))
    end
end)

-----------------------------------------------------------------------------

T("DOT 1D TEST SUBMATRIX", function()
    for i=1,7 do
      local N = 10^i
      local a = matrix(N,2):uniformf(-1,1,rnd):select(2,1) a:update()
      local b = matrix(N,2):uniformf(-1,1,rnd):select(2,1) b:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",a:size(),"1D",
	     measure_process_time(function() a:dot(b) end))
    end
end)

-----------------------------------------------------------------------------

T("CONVOLUTION 2D TEST", function()
    for i=4,6 do
      local N = 6^i
      local a = matrix(root(N,2)):uniformf(-1,1,rnd) a:update()
      local b = matrix(17,17):uniformf(-1,1,rnd) b:update()
      printf("\tsize=%10d  %s  %20.9f  %20.9f\n",a:size(),"1D",
	     measure_process_time(function()
		 matrix.ext.convolution(a, { kernel=b, D=2 })
      end))
    end
end)
