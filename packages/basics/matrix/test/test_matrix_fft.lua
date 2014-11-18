local check = utest.check
local T = utest.test

T("FFTTest", function()
    local sin1 = matrix(512):linspace(0,1*math.pi):sin()
    local sin2 = matrix(512):linspace(0,2*math.pi):sin()
    local sin3 = matrix(512):linspace(0,3*math.pi):sin()
    local sin4 = matrix(512):linspace(0,4*math.pi):sin()
    local sin256 = matrix(512):linspace(0,512*math.pi):sin()
    local sum  = (sin1 + sin2 + sin3 + sin4 + sin256)/5
    local fft  = matrix.ext.real_fftwh(sum, 512, 512):log():clamp(0,math.huge)
    local peaks = fft:gt(3):to_index()
    check.eq( peaks, matrixInt32{1, 2, 3, 4, 256} )
end)
