print("empezamos")
m_xor = matrix.fromString[[
4 3
ascii
0 0 0
0 1 1
1 0 1
1 1 0
]]
ds_input  = dataset.matrix(m_xor,{patternSize={1,2}})
ds_output = dataset.matrix(m_xor,{offset={0,2},patternSize={1,1}})
for i = 1,ds_input:numPatterns() do
  printf("Index %d -> Input: %s Output: %s\n",i,
	 table.concat(ds_input:getPattern(i),","),
	 table.concat(ds_output:getPattern(i),","))
end

function dataset_pair(m,sizein,sizeout)
  local d_in  = dataset.matrix(m,{patternSize = {1,sizein}})
  local d_out = dataset.matrix(m,{offset={0,sizein},patternSize = {1,sizeout}})
  return d_in,d_out
end

ds_input,ds_output = dataset_pair(m_xor,2,1)
for i = 1,ds_input:numPatterns() do
  printf("Index %d -> Input: %s Output: %s\n",i,
	 table.concat(ds_input:getPattern(i),","),
	 table.concat(ds_output:getPattern(i),","))
end
print("finalizamos")
