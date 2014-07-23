local check    = utest.check
local T        = utest.test

local aux = matrix(1,1,3,3):linear()
--print(aux)
local aux = matrix.join(1,aux,aux)
local k = matrix(2,1,2,2):linear()
local o = aux:convolution{ kernel=k, D=2 }
--local o2 = aux:clone("col_major"):convolution{ kernel=k:clone("col_major"), D=2 }

--print(o)
--print(o2)


T("MatrixConvolutionBasicTest",
  function()
    -- from: http://hal.archives-ouvertes.fr/docs/00/11/26/31/PDF/p1038112283956.pdf
    -- High Performance Convolutional Neural Networks for Document Processing
    local m = matrix(1,3,3,3,{ 1,2,0, 1,1,3, 0,2,2, 0,2,1, 0,3,2, 1,1,0, 1,2,1,
                               0,1,3, 3,3,2 })
    local k = matrix(2,3,2,2,{ 1,1, 2,2, 1,1, 1,1, 0,1, 1,0, 1,0, 0,1, 2,1, 2,1,
                               1,2, 2,0, })
    local target_o = matrix(1,2,2,2,{ 14,20, 15,24, 12,24, 17,26, })
    -- replicate the input and target matrices
    local m = matrix.join(1,m,m,m)
    local target_o = matrix.join(1, target_o, target_o, target_o)
    --
    local o  = m:convolution{ kernel=k, D=2 }
    --local o2 = m:clone("col_major"):convolution{ kernel=k:clone("col_major"),
    --D=2 }
    --
    --print(o)
    check.eq(o,  target_o)
    --print(o2)
    --check.eq(o2, target_o:clone("col_major"))
end)

T("MatrixConvolutionMediumTest",
  function()
    local rnd      = random(1234)
    local m        = matrix(2,3,6,6):uniform(-10,10,rnd)
    local k        = matrix(2,3,3,3):uniform(-1,1,rnd)
    local target_o = matrix(2,2,4,4,
                            {
                                -37,  -9, -31,  -3,
                                -9,   0,  13,  13,
                                -6, -38,   1, -25,
                              10,  14,  33, -11,
                              
                                -15, -23, 12,  19,
                                -13,  -8, -5, -28,
                              18,   4, -3, -11,
                              11, -35, -3,  -6,

                              26,  17,   3,  42,
                                -25, -22,  24,  40,
                                -8,  29, -21, -55,
                                -9,  24,  44,  20,

                                -21, -22,  -2,  44,
                              17,  32, -18, -32,
                                -35, -15,  26,   0,                             
                                -28,   3, -38, -11,
    })

    local target_o2 = target_o:clone("col_major")
    local m2 = m:clone("col_major")
    local k2 = k:clone("col_major")
    -------------------------------------------------------------------------
    local o = m:convolution{ kernel=k, D=2 }
    --local o2 = m2:convolution{ kernel=k2, D=2 }
    local c = ann.components.convolution{ kernel = { 3,3,3 }, n=2,
                                          weights = "w1" }
    c:build{ weights = matrix.dict{ w1 = k2:rewrap(2, k:size()/2) } }
    local cnn_o = c:forward(m2):get_matrix()
    --
    check.eq( o, target_o )
    --check.eq( o2, target_o2 )
    check.eq( cnn_o, target_o2 )
end)

if #arg > 0 then
  T("MatrixConvolutionLenaTest",
    function()
      local rnd=random(1234)
      local kx=17
      local ky=17
      local h=10
      -- a matrix of ROWSxCOLUMNSx3
      local m = ImageIO.read(string.get_path(arg[0]) .. "../../../ann/ann/test/photo.png"):
        matrix():transpose():clone()
      local m = m:padding(0,0,8,8,8,8)
      local k = matrix(h,3,kx,ky):uniformf(-0.1,0.1,rnd)
      local o = m:convolution{ kernel=k, D=2 }:squeeze()
      local x,y = o:dim(2),o:dim(3)
      local o = o:rewrap(o:dim(1),x*y)
      local img = ann.connections.input_filters_image(o, {x,y})
      ImageIO.write(img, "wop.png")
      check.TRUE( true )
  end)
end
