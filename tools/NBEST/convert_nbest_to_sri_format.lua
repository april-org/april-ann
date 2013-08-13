fprintf(io.stderr,"# CMD: "..table.concat(arg, " ").."\n")
nbestfilename = arg[1]
outdir        = arg[2]

os.execute("mkdir -p " .. outdir)

nbestf = io.open(nbestfilename, "r")

local i      = 1
local prevn  = nil
local outf   = ""
log10_cte = 1.0/math.log(10)
for line in nbestf:lines() do
  local n,sentence,probs,score = string.match(line, "(.*)|||(.*)|||(.*)|||%s*(.*)%s*")
  probs = string.tokenize(probs, " ")
  if math.mod(i, 1000) then collectgarbage("collect") end
  i = i + 1
  if tonumber(n) ~= tonumber(prevn) then
    printf("%d\n", n) prevn=n
    outf = io.open(outdir .. string.format("/nbest_%06d.score.gz",n), "w")
    --outf:write("NBestList1.0\n")
  end
  --fprintf(outf, "(%s) %s\n", score, sentence)
  for i=1,#probs do
    fprintf(outf, "%f ", tonumber(probs[i]) * log10_cte)
  end
  fprintf(outf, "%s\n", sentence)
end
outf:close()
