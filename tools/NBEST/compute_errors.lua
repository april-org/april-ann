scores  = arg[1]
type    = arg[2]
num_rep = tonumber(arg[3] or 0)
baseline = arg[4]
maxnbest = arg[5]
confidence = tonumber(arg[6] or 0.95)

if maxnbest then maxnbest = tonumber(maxnbest) end

results = {}
sizes   = {}

nscores = 0
score_func = nil

-- calcula la correspondiente funcion
if type == "WER" then
  -- el orden es: Acc Sub Ins Del
  score_func = function(data)
		 return (data[2]+data[3]+data[4]) / (data[1]+data[2]+data[4])
	       end
  nscores = 4
elseif type == "TER" then
  -- el orden es: Errores / Longitud
  score_func = function(data)
		 return data[1] / data[2]
	       end
  nscores = 2
elseif type == "BLEU" then
  -- el orden es: 1gr TGT-1gr 2gr TGT-2gr 3gr TGT-3gr 4gr TGT-4gr 1gr-REF
  score_func = function(data)
		 local logBP = 0
		 if data[2] <= data[9] then
		   logBP = 1 - data[9] / data[2]
		 end
		 local b1 = math.log(data[1]) - math.log(data[2])
		 local b2 = math.log(data[3]) - math.log(data[4])
		 local b3 = math.log(data[5]) - math.log(data[6])
		 local b4 = math.log(data[7]) - math.log(data[8])
		 local logBLEU = (b1 + b2 + b3 + b4)/4 + logBP
		 if data[1] == 0 or data[3] == 0 or data[5] == 0 or data[7] == 0 then
		   return 1
		 end
		 return 1 - math.exp(logBLEU)
	       end  
  nscores = 9
else
  error ("Tipo de score incorrecto: " .. type)
end

fprintf(io.stderr, "Loading data and errors\n")

-- this function loads the scores of each sentence from Nbest lists, keeping the
-- best set of scores for the given objective function
function load_data(filename)
  local f = io.open(filename, "r") or error("File not found: " .. filename)
  local sentences_data = {}
  local results        = {}
  local sizes          = {}
  local n = 1
  while true do
    local line = f:read("*l")
    if not line then break end
    local begin_line = string.tokenize(line)
    assert(begin_line[1] == "SCORES_TXT_BEGIN_0",
	   "Incorrect scores file, header error: '%s'", begin_line[1])
    assert(tonumber(begin_line[4]) == nscores,
	   "Incorrect number of scores")
    local size       = tonumber(begin_line[3])
    local best       = nil
    local bestdata   = nil
    sentences_data[n] = sentences_data[n] or {}
    for i=1,size do
      local data = string.tokenize(f:read("*l"))
      if not maxnbest or i <= maxnbest then
	for i=1,#data do data[i] = tonumber(data[i]) end
	local sc = score_func(data)
	if not best or sc < best then
	  best     = sc
	  bestdata = data
	end
	results[i] = results[i] or {}
	table.insert(results[i], bestdata)
	sizes[i]   = (sizes[i] or 0) + 1
	table.insert(sentences_data[n], bestdata)
      end
    end
    n = n + 1
    local end_line = f:read("*l")
    assert(end_line == "SCORES_TXT_END_0",
	   "Incorrect scores file, footer error: '%s'", end_line)
  end
  return results, sentences_data, sizes
end

function median(points)
  local mpos = math.floor(#points/2)
  local median = points[mpos]
  if #points % 2 ~= 0 then
    median = (median + points[mpos+1])/2
  end
  return median
end

-- this function receives a sorted list of points and computes the confidence
-- interval
function compute_confidence_interval(points, confidence)
  local ic,size = (1 - confidence) / 2, #points
  local pos = math.round(ic * size)
  local a,b = points[pos+1],points[size-pos+1]
  local median = median(points)
  return { median, (b-a)/2 }
end

function overlapped(a1,b1,a2,b2)
  if a1 < a2 and b1 < a2 then return false end
  if a2 < a1 and b2 < a1 then return false end
  return true
end

-- this function receives pair of sorted point lists and computes the 
-- confidence for which the intervals are not overlapped
function compute_significance(p1, p2)
  local p2 = p2 or {}
  if #p2 == 0 then
    for i=1,#p1 do table.insert(p2, 0) end
  end
  assert(#p1 == #p2, "Incorrect list sizes")
  local median_pos = math.round(#p1/2)
  local sz = #p1
  for i=1,median_pos-1 do
    local a1,b1 = p1[i],p1[sz-i+1]
    local a2,b2 = p2[i],p2[sz-i+1]
    if not overlapped(a1,b1,a2,b2) then
      return (sz - i*2) / sz
    end
  end
  return 0.0
end

-- cargamos los datos del target
results,sentences_data,sizes = load_data(scores)
if baseline then
  -- y del baseline en caso de ser necesario
  baseline_results,baseline_sentences_data,baseline_sizes = load_data(baseline)
end

-- compute confidence interval
fprintf(io.stderr, "Computing confidence interval\n")
-- para cosas aleatorias
rnd=random(os.time())

-- tendra el intervalo de confianza del target
local intervals = {}
-- tendra el intervalo de confianza del baseline
local baseline_intervals = {}
-- pairwise en forma de PUNTOS de mejora + intervalo
local pairwise = {}
local pairwise_significance = {}
-- pairwise en forma de PROB de ser mejor a nivel de frase + intervalo
local comparison_probs = {}
-- pairwise en forma de PROB de ser mejor a nivel del conjunto
local total_probs = {}
-- este es para mostrar el tiempo que queda por pantalla
local cronometro = util.stopwatch()
cronometro:go()
-- para cada repeticion
for n=1,num_rep do
  -- variables auxiliares
  local aux          = {}
  local baseline_aux = {}
  local probs        = {}
  local nums = #sentences_data
  -- para el tamanyo de la muestra
  for i=1,nums do
    -- sacamos una frase al azar
    local rand_i = rnd:randInt(1, nums)
    -- para cada N de la lista de NBest
    for j=1,#sentences_data[rand_i] do
      aux[j] = aux[j] or {}
      -- para cada componente
      for k=1,#sentences_data[rand_i][j] do
	-- sumamos los scores de la funcion de error correspondiente a esa
	-- frase, y para cada N de la lista de NBest
	aux[j][k] = (aux[j][k] or 0) + sentences_data[rand_i][j][k]
      end
      -- calculamos la probabilidad de que sea mejor a nivel de frase
      if baseline and baseline_sentences_data[rand_i] and baseline_sentences_data[rand_i][j] then
	probs[j] = probs[j] or 0
	if score_func(sentences_data[rand_i][j]) < score_func(baseline_sentences_data[rand_i][j]) then
	  -- sumamos uno cuando es mejor
	  probs[j] = (probs[j] or 0) + 1
	end
      end
    end
    if baseline then
      -- sumamos los scores del baseline para la funcion de error y esta frase
      for j=1,#baseline_sentences_data[rand_i] do
	baseline_aux[j] = baseline_aux[j] or {}
	for k=1,#baseline_sentences_data[rand_i][j] do
	  baseline_aux[j][k] = (baseline_aux[j][k] or 0) + baseline_sentences_data[rand_i][j][k]
	end
      end
    end
  end
  
  -- calculamos los intervalos y el pairwise
  for i=1,#aux do
    local r = score_func(aux[i])
    intervals[i] = intervals[i] or {}
    table.insert(intervals[i], r)
    if baseline then
      if baseline_aux[i] then
	local base_r = score_func(baseline_aux[i])
	-- pairwise
	pairwise[i] = pairwise[i] or {}
	table.insert(pairwise[i], base_r - r)
	total_probs[i] = total_probs[i] or 0
	if base_r - r > 0 then
	  -- calculamos la probabilidad de ser mejor a nivel de muestra
	  total_probs[i] = total_probs[i] + 1
	end
      end
    end
  end
  if baseline then
    -- calculamos los intervalos y probabilidades del baseline
    for i=1,#baseline_aux do
      -- probabilidad a nivel de frase en esta muestra
      probs[i] = probs[i] / math.min(sizes[i], baseline_sizes[i])
      comparison_probs[i] = comparison_probs[i] or {}
      table.insert(comparison_probs[i], probs[i])
      -- resultado en esta muestra
      local r = score_func(baseline_aux[i])
      baseline_intervals[i] = baseline_intervals[i] or {}
      table.insert(baseline_intervals[i], r)
    end
  end
  -- tiempo estimado
  fprintf(io.stderr, "\rElapsed time: %6d seconds, ETA: %6d seconds",
	  cronometro:read(), cronometro:read()/n * (num_rep-n) )
  io.stderr:flush()
end

local ic = (1 - confidence) / 2

-- calcula los intervalos a partir de las repeticiones
fprintf(io.stderr, "\nSorting intervals\n")
for i=1,#intervals do
  table.sort(intervals[i], function(a,b)
			     return a < b
			   end)
end

if baseline then
  -- lo mismo para el baseline y los diferentes scores de pairwise
  for i=1,#total_probs do
    total_probs[i] = total_probs[i] / num_rep
  end
  for i=1,#baseline_intervals do
    table.sort(baseline_intervals[i], function(a,b)
					return a < b
				      end)
  end
  fprintf(io.stderr, "\nSorting pairwise\n")
  for i=1,#pairwise do
    table.sort(pairwise[i], function(a,b)
			      return a < b
			    end)
    pairwise_significance[i] = compute_significance(pairwise[i])
    pairwise[i] = compute_confidence_interval(pairwise[i], confidence)
  end

  for i=1,#comparison_probs do
    table.sort(comparison_probs[i], function(a,b)
			      return a < b
			    end)
    comparison_probs[i] = compute_confidence_interval(comparison_probs[i],
						      confidence)
  end
end

-- mostramos por pantalla
fprintf(io.stderr, "Computing errors\n")
print("# N  SCORE SCORE-MEDIAN INT BASE BASE-MEDIAN INT PAIRWISE PROBS-SENTENCE PROBS-CORPUS :: SIGNIFICANCE  PWSIGNIFICANCE")
for i=1,#results do -- for each Nbest
  local data = {}
  for j=1,#results[i] do -- for each repetition
    for k=1,#results[i][j] do -- for each score
      data[k] = (data[k] or 0) + results[i][j][k]
    end
  end

  if baseline then
    significance = compute_significance(intervals[i], baseline_intervals[i])
    baseline_intervals[i] = compute_confidence_interval(baseline_intervals[i],
							confidence)
  end
  intervals[i] = compute_confidence_interval(intervals[i], confidence)

  local r = score_func(data)
  local r2 = (intervals[i] or {0,0})[1]
  if type == "BLEU" then r = 1 - r r2 = 1 - r2 end
  printf("%d\t%8.4f %8.4f %8.4f", i, r*100, r2*100, (intervals[i] or {0,0})[2]*100)
  if baseline then
    local data = {}
    for j=1,#baseline_results[i] do
      for k=1,#baseline_results[i][j] do
	data[k] = (data[k] or 0) + baseline_results[i][j][k]
      end
    end
    local r = score_func(data)
    local r2 = (baseline_intervals[i] or {0,0})[1]
    local p = (pairwise[i] or {0,0})[1]
    if type == "BLEU" then r = 1 - r r2 = 1 - r2 end
    printf(" bs: % -10g % -10g % -10g", r*100,
	   r2*100, (baseline_intervals[i] or {0,0})[2]*100)
    printf(" pw: % -10g % -10g", p*100,
	   (pairwise[i] or {0,0})[2]*100)
    printf(" p: % -10g % -10g",
	   (comparison_probs[i] or {0,0})[1]*100,
	   (comparison_probs[i] or {0,0})[2]*100)
    printf(" pW: % -10g", total_probs[i]*100)
    printf(" 1-p-value: % -10g", significance)
    printf(" 1-pw-value: % -10g", pairwise_significance[i])
  end
  printf("\n")
end
