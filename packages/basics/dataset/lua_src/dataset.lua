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

