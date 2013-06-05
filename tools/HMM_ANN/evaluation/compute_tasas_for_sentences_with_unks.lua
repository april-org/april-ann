if #arg ~= 3 then
  printf ("SINTAXIS: %s DICT REFERENCE TARGET\n", string.basename(arg[0]))
  os.exit(1)
end

dict=lexClass.load(io.open(arg[1]))

ref = io.open(arg[2])
tar = io.open(arg[3])

list_unks = {}
list_nounks = {}

local SER = 0
for correcta in ref:lines() do
  local w = string.tokenize(correcta)
  local has_unk = false
  for i=1,#w do
    if dict:getWordId(w[i]) == nil then has_unk = true break end
  end
  local out   = list_nounks
  local recog = tar:read("*l")
  if has_unk then out = list_unks
  elseif recog ~= correcta then
    SER = SER +  1
  end
  table.insert(out, { correcta, recog })
end

ref:close()
tar:close()

local resul_val_WER = tasas{
  typedata = "pairs_lines",
  -- words_sep valor por defecto
  data = list_unks,
  tasa = "ie", -- para calcular wer
}
local resul_val_CER = tasas{
  typedata = "pairs_lines",
  -- words_sep valor por defecto
  data = list_unks,
  tasa = "ie", -- para calcular wer
  words_width = 1,
}

printf ("%7d sentences with    UNKS: WER %5.2f   CER %5.2f\n",
	#list_unks,
	resul_val_WER.tasa,
	resul_val_CER.tasa)



local resul_val_WER = tasas{
  typedata = "pairs_lines",
  -- words_sep valor por defecto
  data = list_nounks,
  tasa = "ie", -- para calcular wer
}
local resul_val_CER = tasas{
  typedata = "pairs_lines",
  -- words_sep valor por defecto
  data = list_nounks,
  tasa = "ie", -- para calcular wer
  words_width = 1,
}

printf ("%7d sentences without UNKS: WER %5.2f   CER %5.2f   SER %5.2f\n",
	#list_nounks,
	resul_val_WER.tasa,
	resul_val_CER.tasa,
	SER/#list_nounks*100)
