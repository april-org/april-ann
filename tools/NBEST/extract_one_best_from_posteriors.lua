nbest = {}
local max = -11111111111111111111

function logAdd(a, b)
  if a>b then
    return a + math.log(1 + math.exp(b-a))
  else
    return b + math.log(1 + math.exp(a-b))
  end
end

if io.read("*l") ~= "NBestList1.0" then error ("Only allow NBestList1.0") end
for line in io.lines() do
  local score,sentence = string.match(line, "^%(([^%(%)]*)%)%s*(.*)%s*$")
  score = tonumber(score)
  if not nbest[sentence] then nbest[sentence] = score
  else nbest[sentence] = logAdd(score, nbest[sentence]) end
  if nbest[sentence] > max then sentmax,max = sentence,nbest[sentence] end
end
printf("%14.6f \t %s\n", max , sentmax)
