stats = stats or {}
stats.MI = stats.MI or {}

local LEVELS  = 256
local EPSILON = 1e-03

-- extracts the histogram from the matrix, using the image package
local function get_histogram_from_matrix(m, levels)
  assert(m, "A matrix is needed")
  -- the matrix is adjusted to be between 0 and 1, and rewrap to fit a image of
  -- only one row
  local maux = m:clone():adjust_range(0.0, 1.0):rewrap(1, m:size())
  local h = image.image_histogram.get_histogram(Image(maux), levels or LEVELS)
  -- sanity check
  assert(math.abs(h:sum() - 1) < EPSILON,
	 "The image_histogram hasn't got bin probabilities")
  return h
end


april_set_doc("stats.MI.entropy",
	      {
		class = "function",
		summary = "Computes the entropy of a given matrix or histogram",
		description = {
		  "Computes the entropy of a given matrix or histogram.",
		  "The two arguments are optional, but one of them must be",
		  "present.",
		  "If the matrix is given (and not the histogram),",
		  "the entropy is computed interpreting the matrix data",
		  "as a continuum sequence of data points, making i.i.d.",
		  "assumptions.",
		},
		params = {
		  "A matrix [optional], ignored if given a histogram",
		  "A histogram matrix [optional]",
		  "The number of histogram bins [optional], by default 256",
		},
		outputs = { "The entropy in base 2" },
	      })

-- computes the entropy given a matrix or a histogram
function stats.MI.entropy(m, histogram, levels)
  assert(not histogram or histogram:is_contiguous(),
	 "Histogram must be contiguous matrix")
  if histogram then
    if m ~= nil then
      fprintf(io.stderr,"If a histogram is given, the matrix will be ignored\n")
    end
    -- we don't know if the histogram has probabilities or raw counts
    local s = histogram:sum() if s~=1.0 then histogram:scal(1/s) end
  end
  local histogram = histogram or get_histogram_from_matrix(m, levels)
  -- the entropy is computed as -sum_i[ p_i * log_2(p_i) ]
  return -histogram:clone():plogp():sum()/math.log(2)
end

april_set_doc("stats.MI.mutual_information",
	      {
		class = "function",
		summary = "Computes the Mutual Information between two matrices",
		description = {
		  "Computes the Mutual Information between two matrices.",
		  "The two matrices must have the same size. The MI is",
		  "computed interpreting the matrix data as a continuum",
		  "sequence of data points, making i.i.d. assumptions.",
		},
		params = {
		  "A matrix",
		  "Another matrix",
		  "The number of histogram bins [optional], by default 256",
		},
		outputs = {
		  "The Mutual Information between given matrices",
		  "The Normalized Mutual Information between given matrices",
		},
	      })

-- this function computes the mutual information between two matrices of the
-- same size
function stats.MI.mutual_information(m1, m2, levels)
  local levels = levels or LEVELS
  assert(m1:size() == m2:size(), "Two matrices must be of the same size")
  -- the matrices are first adjusted to be between 0.0 and levels-1, and
  -- re-wrapped to be a vector
  local m1   = m1:clone():adjust_range(0.0, levels-1):rewrap(m1:size())
  local m2   = m2:clone():adjust_range(0.0, levels-1):rewrap(m2:size())
  -- the joint histogram is a 2-dimensional matrix
  local h12  = matrix(levels,levels):zeros()
  local sz   = m1:size() -- the two matrices has the same size
  -- auxiliary function which computes the bin of a given matrix value
  function get_bin_idx(v) return math.floor(v)+1 end
  for i=1,sz do
    local i1  = get_bin_idx(m1:get(i)) -- bin row position
    local i2  = get_bin_idx(m2:get(i)) -- bin col position
    local c   = h12:get(i1, i2) -- the previous count
    h12:set(i1, i2, c + 1)
  end
  h12:scal(1/sz)
  -- marginalization for m1 and m2 histograms computation
  local h1 = matrix(levels):zeros()
  local h2 = matrix(levels):zeros()
  for i=1,levels do
    h1:axpy(1.0, h12:select(2, i)) -- by columns
    h2:axpy(1.0, h12:select(1, i)) -- by rows
  end
  local entropy1  = stats.MI.entropy(nil, h1)
  local entropy2  = stats.MI.entropy(nil, h2)
  local entropy12 = stats.MI.entropy(nil, h12)
  -- returns the Mutual Information and Normalized Mutual Information
  return entropy1+entropy2-entropy12 , (entropy1+entropy2)/entropy12
end
