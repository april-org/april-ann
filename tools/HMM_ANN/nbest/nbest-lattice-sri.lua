scale=arg[1]
nbestfilename=arg[2]

local outdir=os.tmpname()
os.execute("rm -f " .. outdir)
os.execute("mkdir " .. outdir)

nbestf = io.open(nbestfilename, "r")

fprintf(io.stderr, "GSF: %f\n", scale)
local i      = 1
local prevn  = nil
local outf   = nil
for line in nbestf:lines() do
  local n,sentence,score = string.match(line, "(.*)|||(.*)|||.*|||%s*(.*)%s*")
  if i % 1000 == 0 then collectgarbage("collect") end
  i = i + 1
  if tonumber(n) ~= tonumber(prevn) then
    fprintf(io.stderr, "Converting to SRI format nbest list %d\n", n) prevn=n
    if outf then outf:close() end
    outf = io.open(outdir .. string.format("/nbest_%06d.txt",n), "w")
    outf:write("NBestList1.0\n")
  end
  fprintf(outf, "(%s) %s\n", score, sentence)
end
outf:close()

local auxlist=os.tmpname()
os.execute("ls ".. outdir .. "/* > " .. auxlist)
-- -no-reorder -prime-lattice
local f = io.popen("/home/experimentos/HERRAMIENTAS/bin/nbest-lattice "..
		   " -lattice-wer -nbest-files "..
		   auxlist .. " -rescore-lmw 1 -posterior-scale " .. scale)
fprintf(io.stderr, "Minimizing WER\n")
for line in f:lines() do
  local t = string.tokenize(line)
  fprintf(io.stderr, ".")
  print(table.concat(t, " ", 2, #t))
  io.stderr:flush()
  io.stdout:flush()
end
fprintf(io.stderr, "\n")

os.execute("rm -Rf " .. auxlist .. " " .. outdir)
