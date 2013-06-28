-- OVERWRITTING TOSTRING FUNCTION
matrix.meta_instance.__tostring = function(self)
  local t      = self:toTable()
  local dims   = self:dim()
  local major  = self:get_major_order()
  local coords = {}
  local out    = {}
  local row    = {}
  for i=1,#dims do coords[i]=1 end
  for i=1,#t do
    if #dims > 2 and coords[#dims] == 1 and coords[#dims-1] == 1 then
      table.insert(out,
		   string.format("\n# pos [%s]",
				 table.concat(coords, ",")))
    end
    local j=#dims+1
    repeat
      j=j-1
      coords[j] = coords[j] + 1
      if coords[j] > dims[j] then coords[j] = 1 end
    until j==1 or coords[j] ~= 1
    table.insert(row, string.format("%g", t[i]))
    if coords[#coords] == 1 then
      table.insert(out, table.concat(row, " ")) row={}
    end
  end
  table.insert(out, string.format("# Matrix of size [%s] in %s [%s]",
				  table.concat(dims, ","), major,
				  self:get_reference_string()))
  return table.concat(out, "\n")
end

matrix.meta_instance.__eq = function(op1, op2)
  return op1:equals(op2)
end

matrix.meta_instance.__add = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  return op1:add(op2)
end

matrix.meta_instance.__sub = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  return op1:sub(op2)
end

matrix.meta_instance.__mul = function(op1, op2)
  if not isa(op1,matrix) then op1,op2=op2,op1 end
  if type(op2) == "number" then return op1:scal(op2)
  else return op1:mul(op2)
  end
end

matrix.meta_instance.__pow = function(op1, op2)
  local new_mat = op1:clone()
  return new_mat:pow(op2)
end

matrix.meta_instance.__div = function(op1, op2)
  assert(type(op2) == "number", "Expected a number as second argument")
  local new_mat = op1:clone()
  return new_mat:scal(1/op2)
end

matrix.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
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

function matrix.loadfile(filename)
  local f = io.open(filename,"r") or error("Unable to open " .. filename)
  local b = f:read("*a")
  f:close()
  return matrix.fromString(b)
end

function matrix.savefile(matrix,filename,format)
  local f = io.open(filename,"w") or error("Unable to open " .. filename)
  f:write(matrix:toString(format or "binary"))
  f:close()
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

---------------------------
-- BINDING DOCUMENTATION --
---------------------------
april_set_doc("matrix", {
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

april_set_doc("matrix.__call", {
		class = "method", summary = "Constructor",
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

april_set_doc("matrix.col_major", {
		class = "function", summary = "constructor",
		description ={
		  "Constructor of a multidimensional matrix.",
		  "The data is stored at col_major order, but from",
		  "outside is viewed as row_major (for compatibility",
		  "purposes).",
		},
		params = {
		  "First dimension size",
		  "Second dimension size",
		  "...",
		  "ith dimension size",
		  "...",
		  "nth dimension size",
		  { "A table with values [optional]. The values must be",
		    "in row major order", }
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.fromFilename", {
		class = "function", summary = "constructor",
		description ={
		  "Loads a matrix from a filename.",
		},
		params = {
		  "A filename path.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.toFilename", {
		class = "method",
		summary = "It allows to store a matrix in a file.",
		description ={
		  "It allows to store a matrix in a file.",
		  "It uses the format expected by fromMatrix function.",
		},
		params = {
		  "A filename path.",
		  { "An string with the format: ascii or binary [optional].",
		    "By default is ascii." },
		}, })

april_set_doc("matrix.loadfile", {
		class = "function", summary = "constructor",
		description ={
		  "Loads a matrix from a filename.",
		  "This function supports GZ files, but needs",
		  "to load the matrix as string before parsing it.",
		},
		params = {
		  "A filename path.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.savefile", {
		class = "function",
		summary = "It allows to store a matrix in a file.",
		description ={
		  "It allows to store a matrix in a file.",
		  "This function supports GZ files, but needs",
		  "to save the matrix as string before parsing it.",
		},
		params = {
		  "A matrix object.",
		  "A filename path.",
		  { "An string with the format: ascii or binary [optional].",
		    "By default is binary." },
		}, })

april_set_doc("matrix.loadImage", {
		class = "function", summary = "constructor",
		description ={
		  "Loads a matrix from a image filename.",
		},
		params = {
		  "A filename path.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.saveImage", {
		class = "function",
		summary = "It allows to store a matrix in a image file.",
		description ={
		  "It allows to store a matrix in a file.",
		},
		params = {
		  "A matrix object.",
		  "A filename path.",
		}, })

april_set_doc("matrix.fromString", {
		class = "function", summary = "constructor",
		description ={
		  "Loads a matrix from a Lua string.",
		},
		params = {
		  "A Lua string.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.toString", {
		class = "method",
		summary = "It returns a Lua string which stores the matrix.",
		description ={
		  "It returns a Lua string which stores the matrix.",
		  "It uses the format expected by fromString function.",
		},
		outputs = { "A Lua string" }, })

april_set_doc("matrix.fromPNM", {
		class = "function", summary = "constructor",
		description ={
		  "Loads a matrix from a PNM image stored at a Lua string.",
		},
		params = {
		  "A Lua string.",
		},
		outputs = { "A matrix instantiated object" }, })

april_set_doc("matrix.toPNM", {
		class = "method",
		summary = "It stores the matrix as a PNM image Lua string.",
		description ={
		  "It stores the matrix as a PNM image Lua string.",
		},
		outputs = { "A Lua string" }, })

april_set_doc("matrix.copy_from_table", {
		class = "method",
		summary = "Copies the table values to the matrix.",
		params = {
		  "A lua table with data numbers in row_major order",
		}, })

april_set_doc("matrix.get", {
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

april_set_doc("matrix.set", {
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

april_set_doc("matrix.raw_get", {
		class = "method",
		summary = "Returns the value stored at a given RAW position.",
		params = {
		  "RAW position",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc("matrix.raw_set", {
		class = "method",
		summary = "Sets the value of a given RAW position.",
		params = {
		  "RAW position", 
		},
		outputs = { "The caller matrix" }, })

april_set_doc("matrix.get_offset", {
		class = "method",
		summary = "Returns the RAW offset position of matrix data.",
		outputs = {
		  "A number with the RAW offset position",
		}, })

april_set_doc("matrix.fill", {
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

april_set_doc("matrix.set_use_cuda", {
		class = "method",
		summary = "Indicates if use or not CUDA for math operations.",
		params = {
		  "A boolean",
		},
		outputs = {
		  "The caller object (itself)",
		}, })

april_set_doc("matrix.get_major_order", {
		class = "method",
		summary = "Returns the major order of internal data.",
		outputs = {
		  "A string with the major order",
		}, })

april_set_doc("matrix.dim", {
		class = "method",
		summary = "Returns a table with the size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc("matrix.dim", {
		class = "method",
		summary = "Returns the size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc("matrix.stride", {
		class = "method",
		summary = "Returns a table with the stride size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc("matrix.stride", {
		class = "method",
		summary = "Returns the stride size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc("matrix.slice", {
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

april_set_doc("matrix.clone", {
		class = "method",
		summary = "Returns a deep copy (clone) of the caller matrix.",
		description = {
		  "Returns a deep copy (clone) of the caller matrix.",
		  "It has the possibility of indicate the major order,",
		  "and the data will be reordered if necessary.",
		},
		params = {
		  { "A string: col_major or row_major [optional]. By",
		    "default it is the same major order as the caller matrix" },
		},
		outputs = {
		  "A matrix object (cloned)",
		}, })

april_set_doc("matrix.transpose", {
		class = "method",
		summary = "Returns transposition of the caller matrix.",
		description = {
		  "Returns transposition of the caller matrix.",
		  "The returned matrix is totally new.",
		  "This method is only allowed for 2D matrices",
		},
		outputs = {
		  "A matrix object (transposed)",
		}, })

april_set_doc("matrix.adjust_range", {
		class = "method",
		summary = "Modifies the matrix values IN-PLACE to be at given range",
		params = {
		  "The min value of the range",
		  "The max value of the range"
		},
		outputs = {
		  "The caller matrix object (itself)",
		}, })

april_set_doc("matrix.diag", {
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

april_set_doc("matrix.toTable", {
		class = "method",
		summary = "Returns a Lua table with the data of the matrix.",
		description = {
		  "Returns a Lua table with the data of the matrix.",
		  "The table is a copy of the data in row_major order.",
		},
		outputs = {
		  "A Lua table",
		}, })

april_set_doc("matrix.min", {
		class = "method",
		summary = "Returns the minimum value contained at the matrix.",
		outputs = {
		  "A number",
		}, })

april_set_doc("matrix.max", {
		class = "method",
		summary = "Returns the maximum value contained at the matrix.",
		outputs = {
		  "A number",
		}, })

april_set_doc("matrix.clamp", {
		class = "method",
		summary = "Clamp matrix values IN-PLACE to be in the given range.",
		params = {
		  "The min value of the range",
		  "The max value of the range",
		},
		outputs = {
		  "The caller matrix (itself)",
		}, })

april_set_doc("matrix.add", {
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

april_set_doc("matrix.sub", {
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

april_set_doc("matrix.mul", {
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

april_set_doc("matrix.cmul", {
		class = "method",
		summary = "Returns the component-wise multiplication of caller and other matrix.",
		description = {
		  "Returns the component-wise multiplication of caller and other matrix.",
		  "This method only works with contiguous matrices, which",
		  "are reinterpreted as a vector.",
		  "The returned matrix has only one dimension, however you",
		  "can rewrap it to a different dimension sizes.",
		  "It uses BLAS operations (sbmv).",
		},
		params = {
		  "Another matrix",
		},
		outputs = {
		  "A new matrix result of component-wise multiplication",
		}, })

april_set_doc("matrix.log", {
		class = "method",
		summary = "Component wise log operation IN-PLACE: Y = log(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.log1p", {
		class = "method",
		summary = "Component wise log1p operation IN-PLACE: Y = log1p(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.exp", {
		class = "method",
		summary = "Component wise exp operation IN-PLACE: Y = exp(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.tanh", {
		class = "method",
		summary = "Component wise tanh operation IN-PLACE: Y = tanh(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.sqrt", {
		class = "method",
		summary = "Component wise sqrt operation IN-PLACE: Y = sqrt(Y)",
		params = {
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.pow", {
		class = "method",
		summary = "Component wise pow operation IN-PLACE: Y = pow(Y,x)",
		params = {
		  "A number (x)"
		},
		outputs = {
		  "The caller matrix, Y (itself)",
		}, })

april_set_doc("matrix.axpy", {
		class = "method",
		summary = "BLAS AXPY operation IN-PLACE: Y = Y + alpha * X.",
		params = {
		  "Alpha constant",
		  "Another matrix, X",
		},
		outputs = {
		  "The caller matrix (a vector), Y (itself)",
		}, })

april_set_doc("matrix.gemv", {
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

april_set_doc("matrix.gemm", {
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

april_set_doc("matrix.ger", {
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

april_set_doc("matrix.dot", {
		class = "method",
		summary = "BLAS DOT operation IN-PLACE: v = X  dot  Y",
		params = {
		  "Another matrix, Y (a vector)",
		},
		outputs = {
		  "A number with the dot product",
		}, })

april_set_doc("matrix.scal", {
		class = "method",
		summary = "BLAS SCAL operation IN-PLACE.",
		params = {
		  "The scale factor",
		},
		outputs = {
		  "The caller matrix (itself)."
		}, })

april_set_doc("matrix.norm2", {
		class = "method",
		summary = "BLAS NRM2 operation.",
		outputs = {
		  "A number with the norm-2 of caller matrix."
		}, })

april_set_doc("matrix.copy",
	      {
		class = "method",
		summary = "Copy the values from another matrix using BLAS",
		params  = { "A source matrix" },
		outputs = { "The caller matrix instance" },
	      })
