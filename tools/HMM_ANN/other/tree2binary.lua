if #arg < 3 or #arg > 4 then
  print("Usage: tree2binary vocabulary file.tree file.btree [swap_bytes]")
else
  vocabulary = lexClass.load(io.open(arg[1]))
  treemodel = parser.new_one_step.treeModel{
    filename   = arg[2]
  }
  swap_bytes = ((arg[4] or "no") == "yes")
  if swap_bytes then print("Swapping bytes, changing endianism") end
  treemodel:save{
    filename = arg[3],
    vocabulary = vocabulary:getWordVocabulary(), 
    binary   = true,
    swap     = swap_bytes,
  }
end

