function dataset.to_fann_file_format(input_dataset, output_dataset)
  local numpat = input_dataset:numPatterns()
  if (output_dataset:numPatterns() ~= numpat) then
    error("dataset.to_fann_file_format input and output dataset has different number of patterns")
  end
  local resul = {}
  table.insert(resul,string.format("%d %d %d",numpat,
			       input_dataset:patternSize(),
			       output_dataset:patternSize()))
  for i=1,numpat do
    table.insert(resul,table.concat(input_dataset:getPattern(i)," "))
    table.insert(resul,table.concat(output_dataset:getPattern(i)," "))
  end
  return table.concat(resul,"\n")
end

function dataset.create_fann_file(filename, input_dataset, output_dataset)
  local str  = dataset.to_fann_file_format(input_dataset,output_dataset)
  local fich = io.open(filename,"w")
  fich:write(str)
  fich:close()
end

-----------------------------------------------------------------------------

class("dataset.token.lua_filter")

function dataset.token.lua_filter:__call(t)
  local params = get_table_fields({
				    dataset = { mandatory=true },
				    filter  = { mandatory=true,
						type_match="function" },
				  }, t)
  local obj = { ds=params.dataset, filter=params.filter }
  if isa(ds,dataset) then obj.ds = dataset.token_wrapper(obj.ds) end
  return class_instance(obj, self)
end

function dataset.token.lua_filter:numPatterns() return self.ds:numPatterns() end

function dataset.token.lua_filter:patternSize() return self.ds:patternSize() end

function dataset.token.lua_filter:getPattern(idx)
  local output = self.filter( self.ds:getPattern(idx) )
  assert( isa(output,token.base), "The output of the filter must be a token")
  return output
end

function dataset.token.lua_filter:getPatternBunch(idxs)
  local output = self.filter( self.ds:getPatternBunch(idx) )
  assert( isa(output,token.base), "The output of the filter must be a token")
  return output
end
