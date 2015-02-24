-- Beale's function test
local EPOCHS = 100
local TARGET_X = 3
local TARGET_Y = 0.5

local check = utest.check
local T = utest.test
local M = matrix
local clone = util.clone
local range = iterator.range

if autodiff then
  
  local AD = autodiff
  
  T("BealesFunctionTest",
    function()
      local rnd  = random(1234)
      local x, y = M(1,1,{1}), M(1,1,{1})
      local tbl = { x=x, y=y }
      local opts = {
        ann.optimizer.sgd():
          set_option("learning_rate", 0.06):set_option("momentum", 0.4),
        ann.optimizer.asgd():
          set_option("learning_rate", 0.04),
        ann.optimizer.adadelta(),
        ann.optimizer.adagrad():set_option("learning_rate", 0.1),
        ann.optimizer.cg(),
        ann.optimizer.rprop(),
        ann.optimizer.quickprop():
          set_option("learning_rate", 0.04):set_option("epsilon", 0.01),
        ann.optimizer.simplex(),
      }
      local tbls = range(#opts):map(function(i) return clone(tbl) end):table()
      local x, y = AD.matrix("x y")
      local fxy = AD.op.sum( (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2 )
      local aux = table.pack( fxy, AD.diff(fxy, { x, y }) )
      local func = AD.func(aux, { x, y })
      local eval = function(tbl)
        local x,y = tbl.x,tbl.y
        local fxy,df_dx,df_dy = func(x,y)
        return fxy, { x=df_dx, y=df_dy }
      end
      for i=1,EPOCHS do
        local out = {}
        for j=1,#opts do
          local opt = opts[j]
          local tbl = tbls[j]
          local fxy = opt:execute(eval, tbl)
        end
      end
      for _,tbl in ipairs(tbls) do
        check.number_eq(tbl.x:get(1,1), TARGET_X, 0.1)
        check.number_eq(tbl.y:get(1,1), TARGET_Y, 0.2)
      end
  end)

end
