do
  local _,vector_methods = class.find("mathcore.vector")
  local matrix_ctor = {
    float   = matrix,
    double  = matrixDouble,
    complex = matrixComplex,
    char    = matrixChar,
    bool    = matrixBool,
    int32   = matrixInt32,
  }
  -- not guaranteed to reuse the underlying block, it can be a copy
  vector_methods.to_matrix =
    april_doc{
      class="method",
      summary="Converts the vector into a matrix",
      description="Be careful, the underlying memory block is not guaranteed to be newly allocated",
      params={
        "First dimension size [optional]",
        "Second dimension size [optional]",
        "...",
        "Last dimension size [optional]",
      },
      outputs={
        "A matrix instance",
      },
    } ..
    function(self, ...)
      local block = self:to_block()
      local args = table.pack(...)
      table.insert(args, block)
      local m = matrix_ctor[self.dtype](table.unpack(args))
      assert(#m == #block, "Incompatible matrix sizes")
      return m
    end
end

class.extend_metamethod(matrix, "__len", function(self) return self:dim(1) end)
class.extend_metamethod(matrix, "__ipairs",
                        function(self)
                          return function(self,i)
                            i = i+1
                            if i <= #self then return i,self[i] end
                          end, self, 0
end)

class.extend(matrix, "t", matrix.."transpose")

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrix)

matrix.join =
  matrix.__generic__.__make_generic_join__(matrix)

class.extend(matrix, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

-- ADDING PSEUDO-INVERSE METHODcond
class.extend(matrix, "pinv",
             function(self)
               local u,s,vt = self:svd()
               for aux,i in s:iterate() do
                 u:select(2,i):scal(((math.abs(aux)>1e-07) and 1/aux) or 0.0)
               end
               return matrix.as(self):
                 gemm{
                   A       = vt,
                   B       = u,
                   trans_A = true,
                   trans_B = true,
                   alpha   = 1.0,
                   beta    = 0.0,
                 }
end)

matrix.ext.iterate =
  april_doc{
    class = "function",
    summary = "Returns an iterator which traverses a dimension",
    description = {
      "The iterator uses m:select() method to traverse the given",
      "dimension number. The iterator returns the pair pos,slice",
      "where pos is the position inside the dimension and slice",
      "is a matrix with the result of m:select(dim,pos).",
      "Note that slice is reused between different iterations.",
      "Note that slice is a reference to the original matrix, any",
      "change to slice will be reflected into m."
    },
    params = {
      "A matrix instance (any kind of matrix type)",
      "A dimension number [optional], by default it is 1 (row traversal)",
      "Step [optional], by default it is 1. It can be negative",
    },
    outputs = {
      "An instance of iterator class",
    },
  } ..
  function(self,dim,step)
    local dim  = dim or 1
    local step = step or 1
    assert(step ~= 0, "Unable to iterate with step=0")
    local d = self:dim()
    assert(dim > 0 and dim <= #d, "Out-of-bounds dimension number")
    local slice
    return
      iterator(function(state,pos)
          local self,slice,dim,sz = table.unpack(state)
          pos = pos + step
          if pos <= d[dim] and pos >= 1 then
            slice = self:select(dim,pos,slice)
            return pos,slice
               end end,
        {self,slice,dim,d[dim]}, step>0 and 0 or d[dim]+1)
  end

matrix.ext.broadcast =
  april_doc{
    class = "function",
    summary = "Broadcasts an operation over two matrices",
    description = {
      "Similar to scipy broadcasting: http://wiki.scipy.org/EricsBroadcastingDoc ",
      "The operator is called as: func(a,b,...) where 'a' and 'b' are slices",
      "of the given input matrices, and '...' is the given optional variadic",
      "list of arguments.",
    },
    params = {
      "A binary operator which receives two matrices, called as func(a,b,...)",
      "A matrix",
      "Another matrix",
      "A destination matrix [optional], nil by default",
      "Extra arguments to the binary operator [optional]",
    },
    outputs = { {"The given destination matrix or a new allocated",
                 "matrix with the result of the broadcast"} },
  } ..
  function(func, a, b, result, ...)
    if ... then
      local args = table.pack(...)
      func = function(a,b) return func(a,b,table.unpack(args)) end
    end
    return class.of(a).__broadcast__(func,a,b,result)
  end

matrix.__generic__.__make_index_methods__(matrix)

-- static methods which return a new matrix instead of operate in-place
matrix.op = {}
for _,method in ipairs{"adjust_range", "clamp", "cmul", "cinv", "idiv",
                       "plogp", "log", "log1p", "exp",
                       "sqrt", "pow",
                       "tan", "tanh", "atan",
                       "sin", "sinh", "asin", "asinh",
                       "cos", "cosh", "acos", "acosh",
                       "abs", "complement", "sign", "scal", "div" } do
  matrix.op[method] = function(self,...)
    local clone = self:clone()
    return clone[method](clone,...)
  end
end

function matrix.ext.repmat(x, ...)
  local arg = table.pack(...)
  local dim = x:dim()
  local result_dim = {}
  assert(#arg >= #dim, "Underflow given number of dimensions")
  for i=1,#arg do dim[i] = dim[i] or 1 result_dim[i] = dim[i] * arg[i] end
  local x = x:rewrap(table.unpack(dim))
  local ctor = class.of(x)
  local result = ctor(table.unpack(result_dim))
  local result_sw = result:sliding_window{ size=dim, step=dim }
  local mat
  while not result_sw:is_end() do
    mat = result_sw:get_matrix(mat)
    mat:copy(x)
    result_sw:next()
  end
  return result
end

matrix.ext.diag =
  april_doc{
    class = "function",
    summary = "Returns a matrix with diagonal elements of the given matrix",
    params = {
      "A 2D matrix",
      "The k-th diagonal number [optional], by default k=0",
    },
    outputs = {
      "A new matrix instance",
    }
  } ..
  function(m,k)
    local k=k or 0
    local dim = m:dim()
    assert(#dim == 2, "Needs a 2D matrix")
    local N = dim[1]
    assert(dim[2] == N, "Needs a square matrix")
    local get_map
    if k == 0 then
      get_map = function(i) return m:get(i,i) end
    elseif k>0 then
      assert(k < N, "Out-of-bounds k argument")
      get_map = function(i) return m:get(i,i+k) end
    else -- k<0
      assert(k > -N, "Out-of-bounds k argument")
      get_map = function(i) return m:get(i-k,i) end
    end
    local ctor = class.of(m)
    if rawequal(ctor,matrix.sparse) then ctor = matrix end
    return ctor(N-math.abs(k)):linspace():map(get_map)
  end

matrix.ext.triu =
  april_doc{
    class = "function",
    summary = "Returns uppper triangular matrix taken from given matrix",
    params = {
      "A 2D matrix",
      "The start k-th diagonal number [optional], by default k=0",
    },
    outputs = {
      "A new matrix instance",
    }
  } ..
  function(m,k)
    local k=k or 0
    local dim = m:dim()
    local N = dim[1]
    assert(#dim == 2, "Needs a 2D matrix")
    assert(dim[2] == N, "Needs a square matrix")
    assert(k <= 0 or k <  N, "Out-of-bounds k argument")
    assert(k >= 0 or k > -N, "Out-of-bounds k argument")
    local ctor = class.of(m)
    local triu = ctor(table.unpack(dim)):zeros()
    -- for each row
    for i=1,math.min(N,N-k) do
      local cols = { math.max(1,i+k), N }
      triu[{ i, cols }] = m(i, cols)
    end
    return triu
  end

matrix.ext.tril =
  april_doc{
    class = "function",
    summary = "Returns lower triangular matrix taken from given matrix",
    params = {
      "A 2D matrix",
      "The start k-th diagonal number [optional], by default k=0",
    },
    outputs = {
      "A new matrix instance",
    }
  } ..
  function(m,k)
    local k=k or 0
    local dim = m:dim()
    local N = dim[1]
    assert(#dim == 2, "Needs a 2D matrix")
    assert(dim[2] == N, "Needs a square matrix")
    assert(k <= 0 or k <  N, "Out-of-bounds k argument")
    assert(k >= 0 or k > -N, "Out-of-bounds k argument")
    local ctor = class.of(m)
    local triu = ctor(table.unpack(dim)):zeros()
    local j=math.max(1,k+1) -- col number
    -- for each row
    for i=math.max(1,-k+1),N do
      local cols = { 1, math.min(N,j) } j=j+1
      triu[{ i, cols }] = m(i, cols)
    end
    return triu
  end

-- IMAGE

function matrix.loadImage(filename,format)
  fprintf(io.stderr, "WARNING: matrix.loadImage is deprecated\n")
  local f
  -- si la imagen esta en formato netpbm no hace falta convert:
  if string.match(filename,".*%.p[gnp]m$") then
    f = io.open(filename,"r")
  else
    local dest_format = "pnm"
    if format == "gray" then dest_format = "pgm" end
    f = io.popen(string.format("convert %s %s:-",filename,dest_format))
  end
  if not f then
    error(string.format("Error loading image %s'", filename))
  end
  local b = f:read("*a")
  f:close()
  return matrix.fromPNM(b,format)
end

function matrix.saveImage(matrix,filename)
  local f = io.open(filename,"w") or error("Unable to open " .. filename)
  f:write(matrix:toPNM())
  f:close()
end

function matrix.loadfile()
  error("Deprecated, use fromFilename method")
end

function matrix.savefile()
  error("Deprecated, use toFilename method")
end

-- RAW (ASCII)

-- TODO: MEter esta funcion como parametro format de matrix:toString
function matrix.saveRAW(matrix,filename)
  local f = io.open(filename,"w") or error("Unable to open " .. filename)
  local t = matrix:toTable()
  for i=1,table.getn(t) do
    f:write(t[i] .. "\n")
  end
  f:close()
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

matrix.dict = matrix.dict or {}
setmetatable(matrix.dict, { __call =
                              function(self,v)
                                if v then
                                  return type(v) == "table" and v or {v}
                                else
                                  return {}
                                end
                              end
})

local mmap = function(tbl,func,...)
  if type(tbl) == "table" then
    for name,w in pairs(tbl) do func(w,...) end
  else
    func(tbl,...)
  end
  return tbl
end

local mreduce = function(tbl,func,start,...)
  if type(tbl) == "table" then
    local acc = start
    for name,w in pairs(tbl) do acc = func(acc,w,...) end
    return acc
  else
    return func(start,w,...)
  end
end

matrix.dict.iterator = function(tbl,name_match)
  if type(tbl) == "table" then
    local name_match = name_match or ".*"
    return iterator(pairs(tbl)):
    filter(function(name,w) return name:find(name_match) end)
  else
    assert("Needs a table")
  end
end

for _,name in ipairs{ "scal", "fill", "scalar_add", "pow", "clamp", "idiv", "cinv",
                      "zeros", "ones", "plogp", "log", "log1p", "exp", "sqrt",
                      "tan", "tanh", "atan", "atanh",
                      "cos", "cosh", "acos", "acosh",
                      "sin", "sinh", "asin", "asinh",
                      "abs", "complement", "sign", "inv",
                      "prune_subnormal_and_check_normal", } do
  matrix.dict[name] = function(tbl,...)
    return mmap(tbl, function(w, name, ...) w[name](w,...) end, name, ...)
  end
end

matrix.dict.replace = function(tbl1, tbl2)
  assert(type(tbl1) == "table", "Needs a table as 1st argument")
  assert(type(tbl2) == "table", "Needs a table as 2nd argument")
  for name,w1 in pairs(tbl1) do
    local w2 = tbl2[name]
    if w2 then tbl1[name] = w2 end
  end
  return tbl1
end

matrix.dict.clone = function(tbl)
  if type(tbl) == "table" then
    return iterator(pairs(tbl)):
    map(function(name,w) return name,w:clone() end):table()
  else
    return tbl:clone()
  end
end

matrix.dict.clone_only_dims = function(tbl)
  if type(tbl) == "table" then
    return iterator(pairs(tbl)):
    map(function(name,w) return name,matrix.as(w) end):
      table()
  else
    return matrix.as(tbl)
  end
end

matrix.dict.axpy = function(tbl1,value,tbl2)
  assert(type(value) == "number", "Needs a number as 2nd argument")
  if type(tbl1) == "table" and type(tbl2) == "table" then
    for name,w1 in pairs(tbl1) do
      local w2 = april_assert(tbl2[name], "Unable to find key %s", name)
      w1:axpy(value,w2)
    end
  else
    tbl1:axpy(value, tbl2)
  end
  return tbl1
end

matrix.dict.copy = function(tbl1,tbl2)
  if type(tbl1) == "table" and type(tbl2) == "table" then
    for name,w1 in pairs(tbl1) do
      local w2 = april_assert(tbl2[name], "Unable to find key %s", name)
      w1:copy(w2)
    end
  else
    tbl1:copy(tbl2)
  end
  return tbl1
end

matrix.dict.cmul = function(tbl1,tbl2)
  if type(tbl1) == "table" and type(tbl2) == "table" then
    for name,w1 in pairs(tbl1) do
      local w2 = april_assert(tbl2[name], "Unable to find key %s", name)
      w1:cmul(w2)
    end
  else
    tbl1:cmul(tbl2)
  end
  return tbl1
end

matrix.dict.norm2 = function(tbl)
  if type(tbl) == "table" then
    local n2 = mreduce(tbl, function(acc,w) return acc + w:norm2()^2 end, 0)
    return math.sqrt(n2)
  else
    return tbl:norm2()
  end
end

matrix.dict.size = function(tbl)
  return mreduce(tbl, function(acc,w) return acc + w:size() end, 0)
end

matrix.dict.dot = function(tbl1,tbl2)
  if type(tbl1) == "table" and type(tbl2) == "table" then
    local dot=0
    for name,w1 in pairs(tbl1) do
      local w2 = april_assert(tbl2[name], "Unable to find %s key", name)
      dot = dot + w1:dot(w2)
    end
    return dot
  else
    return tbl1:dot(tbl2)
  end
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

util.vector_uint.meta_instance.__tostring = function(self)
  local out = {}
  local sz = self:size()
  local NCOLS = 9
  for i=1,sz do
    if i%NCOLS == 0 then table.insert(out, "\n") end
    table.insert(out, string.format("%8d ",self:get(i)))
  end
  table.insert(out, string.format("\n# vector_uint of size %d",
				  self:size()))
  return table.concat(out, "")
end

util.vector_float.meta_instance.__tostring = function(self)
  local out = {}
  local sz = self:size()
  local NCOLS = 9
  for i=1,sz do
    if i%NCOLS == 0 then table.insert(out, "\n") end
    table.insert(out, string.format("% -13.8g ",self:get(i)))
  end
  table.insert(out, string.format("\n# vector_float of size %d",
				  self:size()))
  return table.concat(out, " ")
end

---------------------------
-- BINDING DOCUMENTATION --
---------------------------
april_set_doc(matrix, {
		class       = "class",
		summary     = "Multidimensional matrix objects",
		description ={
		  "This class represent multidimensional matrices.",
		  "They are used for build datasets and train machine",
		  "learning models.",
		  "Mathematical operations are allowed (*, -, +).",
		  "Specific BLAS methods are binding to Lua to ensure",
		  "efficiency."
		}, })

april_set_doc(matrix, {
		class = "method", summary = "Matrix constructor",
		description ={
		  "Constructor of a multidimensional matrix.",
		  "The data is stored at row_major order",
		},
		params = {
		  "First dimension size",
		  "Second dimension size",
		  "...",
		  "ith dimension size",
		  "...",
		  "nth dimension size",
		  { "A table with values [optional]. The values must be",
		    "in row major order", },
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc(matrix.read, {
		class = "method",
		summary = "It allows to read a matrix from a stream.",
		description ={
		  "It allows to read a matrix from a stream.",
		  "It uses the format of write function.",
		},
		params = {
		  "A aprilio.stream instance.",
                  "A lua table with options",
		}, })

april_set_doc(matrix.fromMMap, {
		class = "function", summary = "Matrix fromMMap constructor",
		description ={
		  "Loads a matrix from a mmaped filename.",
		},
		params = {
		  "A filename path.",
		  {
		    "A boolean indicating if writing is allowed [optional].",
		    "By default it is true",
		  },
		  {
		    "A boolean indicating if memory map is shared [optional].",
		    "By default it is true",
		  },
		  
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc(matrix.."toMMap", {
		class = "method",
		summary = "It allows to store a matrix in a mmapped file.",
		description ={
		  "It allows to store a matrix in a mmapped file.",
		  "It uses the format expected by fromMMap function.",
		},
		params = {
		  "A filename path.",
		}, })

april_set_doc(matrix.loadImage, {
		class = "function", summary = "Matrix loadImage constructor",
		description ={
		  "Loads a matrix from a image filename.",
		},
		params = {
		  "A filename path.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc(matrix.saveImage, {
		class = "function",
		summary = "It allows to store a matrix in a image file.",
		description ={
		  "It allows to store a matrix in a file.",
		},
		params = {
		  "A matrix object.",
		  "A filename path.",
		}, })

april_set_doc(matrix.fromString, {
		class = "function", summary = "Matrix fromString constructor",
		description ={
		  "Loads a matrix from a Lua string.",
		},
		params = {
		  "A Lua string.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc(matrix.."toString", {
		class = "method",
		summary = "It returns a Lua string which stores the matrix.",
		description ={
		  "It returns a Lua string which stores the matrix.",
		  "It uses the format expected by fromString function.",
		},
		outputs = { "A Lua string" }, })

april_set_doc(matrix.."to_lua_string", {
		class = "method",
		summary = "It returns a Lua chunk string which is loadable.",
                params  = { "The format [optional]. By default is binary." },
		outputs = { "A Lua string" }, })

april_set_doc(matrix.fromPNM, {
		class = "function", summary = "Matrix fromPNM constructor",
		description ={
		  "Loads a matrix from a PNM image stored at a Lua string.",
		},
		params = {
		  "A Lua string.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc(matrix.."toPNM", {
		class = "method",
		summary = "It stores the matrix as a PNM image Lua string.",
		description ={
		  "It stores the matrix as a PNM image Lua string.",
		},
		outputs = { "A Lua string" }, })

april_set_doc(matrix.."copy_from_table", {
		class = "method",
		summary = "Copies the table values to the matrix.",
		params = {
		  "A lua table with data numbers in row_major order",
		}, })

april_set_doc(matrix.."get", {
		class = "method",
		summary = "Returns the value stored at a given position.",
		params = {
		  "First dimension position",
		  "Second dimension position",
		  "...",
		  "ith dimension position",
		  "...",
		  "nth dimension position",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."sum",
	      {
		class="method",
		summary="Computes the sum of all the elements.",
		outputs={"A number"},
})

april_set_doc(matrix.."sum",
	      {
		class="method",
		summary="Computes the sum of all the elements over the given dimension.",
		params={
		  "A number, the dimension",
		  "A matrix where to store the result [optional]",
		},
		outputs={"A matrix with the result"},
})

april_set_doc(matrix.."set", {
		class = "method",
		summary = "Sets the value of a given position.",
		params = {
		  "First dimension position",
		  "Second dimension position",
		  "...",
		  "ith dimension position",
		  "...",
		  "nth dimension position",
		  "A number with the value to be set",
		},
		outputs = { "The caller matrix" }, })

april_set_doc(matrix.."raw_get", {
		class = "method",
		summary = "Returns the value stored at a given RAW position.",
		params = {
		  "RAW position",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."raw_set", {
		class = "method",
		summary = "Sets the value of a given RAW position.",
		params = {
		  "RAW position", 
		},
		outputs = { "The caller matrix" }, })

april_set_doc(matrix.."offset", {
		class = "method",
		summary = "Returns the RAW offset position of matrix data.",
		outputs = {
		  "A number with the RAW offset position",
		}, })

april_set_doc(matrix.."fill", {
		class = "method",
		summary = "Sets all values to a given number.",
		description = {
		  "Sets all values to a given number.",
		  "This method modifies the object IN-PLACE.",
		},
		params = {
		  "A number",
		},
		outputs = {
		  "The caller object (itself)",
		}, })

april_set_doc(matrix.."set_use_cuda", {
		class = "method",
		summary = "Indicates if use or not CUDA for math operations.",
		params = {
		  "A boolean",
		},
		outputs = {
		  "The caller object (itself)",
		}, })

april_set_doc(matrix.."dim", {
		class = "method",
		summary = "Returns a table with the size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc(matrix.."dim", {
		class = "method",
		summary = "Returns the size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."stride", {
		class = "method",
		summary = "Returns a table with the stride size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc(matrix.."stride", {
		class = "method",
		summary = "Returns the stride size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."slice", {
		class = "method",
		summary = "Returns a sub-matrix that is a slice of caller matrix.",
		description = {
		  "Returns a sub-matrix that is a slice of caller matrix.",
		  "This method returns a sub-matrix which references the",
		  "parent matrix (not copy the data). Optionally it is possible",
		  "to do a deep copy (clone) of the data.",
		},
		params = {
		  "A table with the first position of the sub-matrix",
		  "A table with the sizes of each dimension for the sub-matrix",
		  { "A boolean indicating if do or not a clone [optional]. By",
		    "default it is set to false", },
		},
		outputs = {
		  "A matrix object (sub-matrix)",
		}, })

april_set_doc(matrix.."rewrap", {
		class = "method",
		summary = "Reinterprets the data as with other dimensions.",
		description = {
		  "Returns a matrix which references to the caller, but",
		  "reinterpreting the internal data with the given array of",
		  "dimension sizes.",
		  "The caller matrix must be a contiguous chunk of data.",
		},
		params = {
		  { "A table with the size of each dimension.",
		    "The number of dimensions could be different of ",
		    "caller matrix.", },
		},
		outputs = {
		  "A matrix object (referencing the caller matrix)",
		}, })

april_set_doc(matrix.."select", {
		class = "method",
		summary = "Returns an slice result of select given dimension at given index.",
		description = {
		  "Returns an slice result of select given dimension at given index.",
		  "The matrix has one less dimension because the selected dimension",
		  "is removed.",
		},
		params = {
		  { "A number with the selected dimension" },
		  { "A number with the selected index" },
		},
		outputs = {
		  "A matrix object (referencing the caller matrix)",
		}, })

april_set_doc(matrix.join, {
		class = "function",
		summary = "Produce a matrix which is the join of a given set of matrices.",
		description = {
		  "Joins a given set of matrices, given the dimension where they differ.",
		  "Be careful, this method duplicates the memory needed, because all the",
		  "matrices will be copied to the destination.",
		},
		params = {
		  { "A number with the dimension where matrices differ." },
		  { "The 1st matrix" },
		  { "The 2nd matrix" },
		  { "..." },
		  { "The Nth matrix" },
		},
		outputs = {
		  "A new matrix object",
		}, })

april_set_doc(matrix.."clone", {
		class = "method",
		summary = "Returns a deep copy (clone) of the caller matrix.",
		outputs = {
		  "A matrix object (cloned)",
		}, })

april_set_doc(matrix.."transpose", {
		class = "method",
		summary = "Returns transposition of the caller matrix.",
		description = {
		  "Returns transposition of the caller matrix.",
		  "The returned matrix is a reference to the original.",
		},
		outputs = {
		  "A matrix object (transposed)",
		}, })

april_set_doc(matrix.."transpose", {
		class = "method",
		summary = "Transposes two dimensions.",
		description = {
		  "Transposes two dimensions of the caller matrix.",
		  "The returned matrix is a reference to the original.",
		},
                params = {
                  "One dimension number",
                  "Second dimension number",
                },
		outputs = {
		  "A matrix object (transposed)",
		}, })

april_set_doc(matrix.."adjust_range", {
		class = "method",
		summary = "Modifies the matrix values IN-PLACE to be at given range",
		params = {
		  "The min value of the range",
		  "The max value of the range"
		},
		outputs = {
		  "The caller matrix object (itself)",
		}, })

april_set_doc(matrix.."diag", {
		class = "method",
		summary = "Sets diagonal positions to a given number.",
		description = {
		  "Sets diagonal positions to a tiven number value.",
		  "This method modifies the object IN-PLACE.",
		},
		params = {
		  "A number",
		},
		outputs = {
		  "The caller object (itself)",
		}, })

april_set_doc(matrix.."toTable", {
		class = "method",
		summary = "Returns a Lua table with the data of the matrix.",
		description = {
		  "Returns a Lua table with the data of the matrix.",
		  "The table is a copy of the data in row_major order.",
		},
		outputs = {
		  "A Lua table",
		}, })

april_set_doc(matrix.."min", {
		class = "method",
		summary = "Returns the minimum value contained at the matrix.",
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."min", {
		class = "method",
		summary = "Returns a matrix with minimum values over given dimension.",
		params = {
		  "A number with the dimension",
		  "A matrix where to store the result [optional]",
		},
		outputs = {
		  "A matrix with the result",
		}, })

april_set_doc(matrix.."max", {
		class = "method",
		summary = "Returns the maximum value contained at the matrix.",
		outputs = {
		  "A number",
		}, })

april_set_doc(matrix.."max", {
		class = "method",
		summary = "Returns a matrix with maximum values over given dimension.",
		params = {
		  "A number with the dimension",
		  "A matrix where to store the result [optional]",
		},
		outputs = {
		  "A matrix with the result",
		}, })

april_set_doc(matrix.."clamp", {
		class = "method",
		summary = "Clamp matrix values IN-PLACE to be in the given range.",
		params = {
		  "The min value of the range",
		  "The max value of the range",
		},
		outputs = {
		  "The caller matrix (itself)",
		}, })

april_set_doc(matrix.."add", {
		class = "method",
		summary = "Returns the addition of caller and other matrix.",
		description = {
		  "Returns a new matrix which is addition of caller and other matrix.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrix",
		},
		outputs = {
		  "A new matrix result of addition",
		}, })

april_set_doc(matrix.."sub", {
		class = "method",
		summary = "Returns the subtraction of caller and other matrix.",
		description = {
		  "Returns a new matrix which is the subtraction of caller and other matrix.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrix",
		},
		outputs = {
		  "A new matrix result of subtraction",
		}, })

april_set_doc(matrix.."mul", {
		class = "method",
		summary = "Returns the multiplication of caller and other matrix.",
		description = {
		  "Returns a new matrix which is the multiplication of caller and other matrix.",
		  "This method works with vector-vector, dot product,",
		  "vector-matrix, and matrix-matrix multiplication, depending",
		  "on the matrices dimensions.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrix (2D matrix or a vector)",
		},
		outputs = {
		  "A new matrix result of multiplication",
		}, })

april_set_doc(matrix.."cmul", {
		class = "method",
		summary = "Computes IN-PLACE the component-wise multiplication of caller and other matrix.",
		params = {
		  "Another matrix",
		},
		outputs = {
		  "The caller matrix",
		}, })

april_set_doc(matrix.."plogp", {
		class = "method",
		summary = "Component wise p*log(p) operation IN-PLACE: Y = Y*log(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."log", {
		class = "method",
		summary = "Component wise log operation IN-PLACE: Y = log(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."log1p", {
		class = "method",
		summary = "Component wise log1p operation IN-PLACE: Y = log1p(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."exp", {
		class = "method",
		summary = "Component wise exp operation IN-PLACE: Y = exp(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."tan", {
		class = "method",
		summary = "Component wise tanh operation IN-PLACE: Y = tan(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."tanh", {
		class = "method",
		summary = "Component wise tanh operation IN-PLACE: Y = tanh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."atan", {
		class = "method",
		summary = "Component wise atanh operation IN-PLACE: Y = atan(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."atanh", {
		class = "method",
		summary = "Component wise atanh operation IN-PLACE: Y = atanh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."sin", {
		class = "method",
		summary = "Component wise sin operation IN-PLACE: Y = sin(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."sinh", {
		class = "method",
		summary = "Component wise sinh operation IN-PLACE: Y = sinh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."asin", {
		class = "method",
		summary = "Component wise asin operation IN-PLACE: Y = asin(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."asinh", {
		class = "method",
		summary = "Component wise asinh operation IN-PLACE: Y = asinh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."cos", {
		class = "method",
		summary = "Component wise cos operation IN-PLACE: Y = cos(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."cosh", {
		class = "method",
		summary = "Component wise cosh operation IN-PLACE: Y = cosh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."acos", {
		class = "method",
		summary = "Component wise acos operation IN-PLACE: Y = acos(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."acosh", {
		class = "method",
		summary = "Component wise acosh operation IN-PLACE: Y = acosh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."abs", {
		class = "method",
		summary = "Component wise abs operation IN-PLACE: Y = abs(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."complement", {
		class = "method",
		summary = "Component wise complement operation IN-PLACE: Y = 1 - Y",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."sqrt", {
		class = "method",
		summary = "Component wise sqrt operation IN-PLACE: Y = sqrt(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."pow", {
		class = "method",
		summary = "Component wise pow operation IN-PLACE: Y = pow(Y,x)",
		params = {
		  "A number (x)"
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc(matrix.."axpy", {
		class = "method",
		summary = "BLAS AXPY operation IN-PLACE: Y = Y + alpha * X.",
		params = {
		  "Alpha constant",
		  "Another matrix, X",
		},
		outputs = {
		  "The caller matrix (a vector), Y (itself)",
		}, })

april_set_doc(matrix.."gemv", {
		class = "method",
		summary = "BLAS GEMV operation IN-PLACE: Y = beta * Y + alpha * A * X.",
		params = {
		  ["alpha"] = "Alpha constant",
		  ["beta"]  = "Beta constant",
		  ["trans_A"]  = {
		    "A boolean indicating if transpose or not matrix A.",
		    "It is [optional], by default is false",
		  },
		  ["A"] = "Another matrix, A (a 2D matrix)",
		  ["X"] = "Another matrix, X (a vector)",
		},
		outputs = {
		  "The caller matrix (a vector), Y (itself)",
		}, })

april_set_doc(matrix.."gemm", {
		class = "method",
		summary = "BLAS GEMM operation IN-PLACE: C = beta * C + alpha * A * B.",
		params = {
		  ["alpha"] = "Alpha constant",
		  ["beta"]  = "Beta constant",
		  ["trans_A"]  = {
		    "A boolean indicating if transpose or not matrix A.",
		    "It is [optional], by default is false",
		  },
		  ["trans_B"]  = {
		    "A boolean indicating if transpose or not matrix B.",
		    "It is [optional], by default is false",
		  },
		  ["A"] = "Another matrix, A (a 2D matrix)",
		  ["B"] = "Another matrix, B (a 2D matrix)",
		},
		outputs = {
		  "The caller matrix, C (itself)",
		}, })

april_set_doc(matrix.."ger", {
		class = "method",
		summary = "BLAS GER operation IN-PLACE: Z = Z + alpha * X * Y'.",
		params = {
		  ["alpha"] = "Alpha constant",
		  ["X"] = "Another matrix, X (a vector)",
		  ["Y"] = "Another matrix, Y (a vector)",
		},
		outputs = {
		  "The caller matrix, Z (itself)",
		}, })

april_set_doc(matrix.."dot", {
		class = "method",
		summary = "BLAS DOT operation IN-PLACE: v = X  dot  Y",
		params = {
		  "Another matrix, Y (a vector)",
		},
		outputs = {
		  "A number with the dot product",
		}, })

april_set_doc(matrix.."scal", {
		class = "method",
		summary = "BLAS SCAL operation IN-PLACE.",
		params = {
		  "The scale factor",
		},
		outputs = {
		  "The caller matrix (itself)."
		}, })

april_set_doc(matrix.."div", {
		class = "method",
		summary = "Inverse to scal operation IN-PLACE, produces A = value/A.",
		params = {
		  "The div factor",
		},
		outputs = {
		  "The caller matrix (itself)."
		}, })

april_set_doc(matrix.."norm2", {
		class = "method",
		summary = "BLAS NRM2 operation.",
		outputs = {
		  "A number with the norm-2 of caller matrix."
		}, })

april_set_doc(matrix.."copy",
	      {
		class = "method",
		summary = "Copy the values from another matrix using BLAS",
		params  = { "A source matrix" },
		outputs = { "The caller matrix instance" },
})

april_set_doc(matrix.."linear",
	      {
		class = "method",
		summary = "Initializes with linear integers",
		params  = { "First integer value [optional], by default 0",
			    "Step value [optional], by default 1", },
		outputs = { "The caller matrix instance" },
})

april_set_doc(matrix.."uniform",
	      {
		class = "method",
		summary = "Initializes with random positive integers from range [a,b]",
		params  = { "Lower range value",
			    "Upper range value",
			    "A random object instance [optional]" },
		outputs = { "The caller matrix instance" },
})

april_set_doc(matrix.."uniformf",
	      {
		class = "method",
		summary = "Initializes with random floats in range [a,b]",
		params  = { "Lower range value",
			    "Upper range value",
			    "A random object instance [optional]" },
		outputs = { "The caller matrix instance" },
})

april_set_doc(matrix.."is_contiguous",
	      {
		class = "method",
		summary = "Returns true if the matrix data is contiguous at memory",
		outputs = { "A boolean" },
})

april_set_doc(matrix.."scalar_add",
	      {
		class = "method",
		summary = "Adds a scalar IN-PLACE",
		params  = { "A number" },
		outputs = { "The caller matrix instance" },
})

april_set_doc(matrix.."inv",
	      {
		class = "method",
		summary = "Computes the inverse of a matrix",
		description = {
		  "This method computes the inverse of matrix.",
		  "Check that your matrix is not singular, otherwise",
		  "the returned matrix won't be correct.",
		},
		outputs = { "The matrix inverse" },
})

april_set_doc(matrix.."svd",
	      {
		class = "method",
		summary = "Computes the SVD of a matrix",
		description = {
		  "This method computes the SVD of matrix.",
                  "The computation returns three matrices",
		  ", so A=U * S * V'.",
		},
		outputs = {
		  "The matrix U",
		  "The sparse row vector S with the eigenvalues",
		  "The matrix V', the transposed of V",
		},
})

april_set_doc(matrix.."diagonalize",
	      {
		class = "method",
		summary = "Converts the given uni-dimensional matrix in a bi-dimensional diagonal dense matrix",
		outputs = {
		  "A matrix which is the diagonalized version of the caller matrix",
		},
})

april_set_doc(matrix.."contiguous",
	      {
		class = "method",
		summary = "Returns a contiguous version of the caller matrix",
		description = {
		  "Returns a contiguous version of the caller matrix.",
		  "If the matrix is contiguous, returns itself.",
		  "Otherwise, returns a copy of the caller.",
		},
		outputs = { "A matrix instance" },
})

april_set_doc(matrix.."map",
	      {
		class = "method",
		summary = "Maps the matrix values by a given list of matrices and a Lua map function",
		description = {
		  "Maps the matrix values by a given list of matrices",
		  "and a Lua map function.",
		  "The Lua function will be called for every possible",
		  "matrix position. The Lua function receives the caller matrix",
		  "value at the given position, the value of the second matrix,",
		  "the value of the third matrix, and so on.",
		  "The Lua function returns NIL or ONLY one value, which will be",
		  "assigned to the caller matrix IN-PLACE.",
		  "All the matrices must have the same dimension sizes.",
		  "The number of given matrices could be >= 0",
		},
		params = {
		  "A second matrix",
		  "A third matrix",
		  "...",
		  "A Nth matrix",
		  "A Lua function which applies the map computation.",
		},
		outputs = { "The caller matrix" },
})

april_set_doc(matrix.."lt",
	      {
		class = "method",
		summary = "Returns a matrixBool with true where values are less than given param.",
		params = {
		  "A matrix or a number",
		},
		outputs = { "A matrixBool instance" },
})

april_set_doc(matrix.."gt",
	      {
		class = "method",
		summary = "Returns a  matrixBool with true where values are greater than given param.",
		params = {
		  "A matrix or a number",
		},
		outputs = { "A matrixBool instance" },
})

-------------------------------------------------------------------------

april_set_doc(matrix.."sliding_window",
	      {
		class = "method",
		summary = "Returns a sliding window object",
		params  = {
		  offset = "Lua table [optional]",
		  size = "Lua table [optional]",
		  step = "Lua table [optional]",
		  numSteps = "Lua table [optional]",
		  orderStep = "Lua table [optional]",
		},
		outputs = { "An instance of matrix.__sliding_window__" },
})

april_set_doc(matrix.__sliding_window__,
	      {
		class       = "class",
		summary     = "Sliding window for matrix objects",
})

april_set_doc(matrix.__sliding_window__.."get_matrix",
	      {
		class       = "method",
		summary     = "Returns the matrix generated by the window",
		params      = { {"A matrix [optional], to be used instead",
				 "of alloc a new one"} },
		outputs     = { "A matrix instance" },
})

april_set_doc(matrix.__sliding_window__.."next",
	      {
		class       = "method",
		summary     = "Moves the window to the next position",
		outputs     = { "The caller sliding_window object" },
})

april_set_doc(matrix.__sliding_window__.."is_end",
	      {
		class       = "method",
		summary     = "Returns true when it finishes",
		outputs     = { "A boolean" },
})

april_set_doc(matrix.__sliding_window__.."iterate",
	      {
		class       = "method",
		summary     = "Returns an iterator function: for mat in s:iterate() do ... end",
		outputs     = { "An iterator function" },
})

-----------------------------
-- DEPRECATED CONSTRUCTORS --
matrix.row_major = make_deprecated_function("matrix.row_major", "matrix", matrix)
matrix.col_major = make_deprecated_function("matrix.col_major", "matrix", matrix)
class.extend(matrix, "get_major_order",
             make_deprecated_function("matrix.get_major_order", nil,
                                      function(self) return "row_major" end))
-----------------------------

-- other stuff

matrix.meta_instance.__call =
  matrix.__generic__.__make_generic_call__()

-- define left side operator []
matrix.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrix)

-- define right side operator []
matrix.__generic__.__make_generic_index__(matrix)

matrix.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("Matrix",
                                            function(value)
                                              return string.format("% -13.6g", value)
  end)

matrix.meta_instance.__eq = function(op1, op2)
  if type(op1) == "number" or type(op2) == "number" then return false end
  return op1:equals(op2)
end

matrix.meta_instance.__add = function(op1, op2)
  if not class.is_a(op1,matrix) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scalar_add(op2)
  else
    return op1:add(op2)
  end
end

matrix.meta_instance.__sub = function(op1, op2)
  if class.is_a(op1,matrix) and class.is_a(op2,matrix) then
    return op1:sub(op2)
  elseif class.is_a(op1,matrix) then
    return op1:clone():scalar_add(-op2)
  elseif class.is_a(op2,matrix) then
    return op2:clone():scal(-1):scalar_add(op1)
  end
end

matrix.meta_instance.__mul = function(op1, op2)
  if class.is_a(op1,matrix.sparse) or class.is_a(op2,matrix.sparse) then
    if class.is_a(op2,matrix.sparse) then
      local res = matrix(op1:dim(1),op2:dim(2))
      res:transpose():sparse_mm{ alpha=1.0, beta=0.0, A=op2, B=op1,
                                 trans_A=true, trans_B=true }
      return res
    else
      local res = matrix(op1:dim(1),op2:dim(2))
      res:sparse_mm{ alpha=1.0, beta=0.0, A=op1, B=op2 }
      return res
    end
  else
    if not class.is_a(op1,matrix) then op1,op2=op2,op1 end
    if type(op2) == "number" then return op1:clone():scal(op2)
    else return op1:mul(op2)
    end
  end
end

matrix.meta_instance.__pow = function(op1, op2)
  local new_mat = op1:clone()
  return new_mat:pow(op2)
end

matrix.meta_instance.__div = function(op1, op2)
  if type(op2) == "number" then
    local new_mat = op1:clone()
    return new_mat:scal(1/op2)
  elseif type(op1) == "number" then
    local new_mat = op2:clone()
    return new_mat:div(op1)
  else
    assert(class.is_a(op1,matrix) and class.is_a(op2,matrix),
	   "Expected a matrix and a number or two matrices")
    local new_mat1 = op1:clone()
    local new_mat2 = op2:clone():div(1)
    return new_mat1:cmul(new_mat2)
  end
end

matrix.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end
