local check = utest.check
local w = matrix.col_major(4,3):uniformf(0,1,random(1234))
local input = matrix.col_major(5,3,{ 0, 1, 0,
                                     1, 1, 1,
                                     1, 0, 1,
                                     1, 0, 0,
                                     0, 0, 1 })
local sparse_input = matrix.sparse.csr(input)
local e = matrix.col_major(5,4):uniformf(0,1,random(2384))
--
for w,transpose in iterator(ipairs({ {w,false},{w:transpose():clone(),true} })):
select(2):map(table.unpack):get() do
  local c = ann.components.dot_product{
    input = 3,
    output = 4,
    weights = "w",
    transpose = transpose,
  }:build{ weights=matrix.dict{ w=w } }
  --
  local output = c:forward(input):get_matrix()
  c:backprop(e)
  local grads1 = c:compute_gradients()
  --
  local sparse_output = c:forward(sparse_input):get_matrix()
  c:backprop(e)
  local grads2 = c:compute_gradients()
  --
  check.eq(output,sparse_output)
  check.eq(grads1("w"),grads2("w"))
end
