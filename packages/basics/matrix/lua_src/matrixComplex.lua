class.extend_metamethod(matrixComplex, "__len", function(self) return self:dim(1) end)
class.extend_metamethod(matrixComplex, "__ipairs",
                        function(self)
                          return function(self,i)
                            i = i+1
                            if i <= #self then return i,self[i] end
                          end, self, 0
end)

class.extend(matrixComplex, "t", matrixComplex.."transpose")

class.extend(matrixComplex, "flatten",
             function(self)
               return self:rewrap(self:size())
end)

matrix.__generic__.__make_index_methods__(matrixComplex)

-- serialization
matrix.__generic__.__make_all_serialization_methods__(matrixComplex)

matrixComplex.join =
  matrix.__generic__.__make_generic_join__(matrixComplex)

matrixComplex.meta_instance.__eq = function(op1, op2)
  return op1:equals(op2)
end

matrixComplex.meta_instance.__add = function(op1, op2)
  if not class.is_a(op1,matrixComplex) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scalar_add(op2)
  else
    return op1:add(op2)
  end
end

matrixComplex.meta_instance.__sub = function(op1, op2)
  if not class.is_a(op1,matrixComplex) then op1,op2=op2,op1 end
  return op1:sub(op2)
end

matrixComplex.meta_instance.__mul = function(op1, op2)
  if not class.is_a(op1,matrixComplex) then op1,op2=op2,op1 end
  if type(op2) == "number" then
    return op1:clone():scal(complex(op2,0))
  elseif type(op2) == "string" then
    return op1:clone():scal(complex(op2))
  elseif class.is_a(op2,complex) then
    return op1:clone():scal(op2)
  else return op1:mul(op2)
  end
end

matrixComplex.meta_instance.__div = function(op1, op2)
  assert(type(op2) == "number", "Expected a number as second argument")
  local new_mat = op1:clone()
  return new_mat:scal(1/op2)
end

matrixComplex.meta_instance.__unm = function(op)
  local new_mat = op:clone()
  return new_mat:scal(-1)
end

-----------------------------------------------------------------------------

---------------------------
-- BINDING DOCUMENTATION --
---------------------------
april_set_doc(matrixComplex, {
		class       = "class",
		summary     = "Multidimensional matrixComplex objects",
		description ={
		  "This class represent multidimensional matrices as matrix.",
		  "But this is not available for build datasets and train machine",
		  "learning models. In the last instance, you need always to",
		  "transform this to a vanilla matrix.",
		  "Mathematical operations are allowed (*, -, +).",
		  "Specific BLAS methods are binding to Lua to ensure",
		  "efficiency."
		}, })

april_set_doc(matrixComplex, {
		class = "method", summary = "Constructor",
		description ={
		  "Constructor of a multidimensional matrixComplex.",
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
		    "in row major order. A valid value is a number, which is",
		    "taken as real part only, a complex object instance, or a",
		    "string with the complex number.", },
		},
		outputs = { "A matrixComplex instantiated object" }, })

april_set_doc(matrixComplex.."to_lua_string", {
		class = "method",
		summary = "It returns a Lua chunk string which is loadable.",
                params  = { "The format [optional]. By default is binary." },
		outputs = { "A Lua string" }, })


april_set_doc(matrixComplex.."copy_from_table", {
		class = "method",
		summary = "Copies the table values to the matrixComplex.",
		params = {
		  "A lua table with data numbers in row_major order",
		}, })

april_set_doc(matrixComplex.."sum",
	      {
		class="method",
		summary="Computes the sum of all the elements.",
		outputs={"A number"},
})

april_set_doc(matrixComplex.."sum",
	      {
		class="method",
		summary="Computes the sum of all the elements over the given dimension.",
		params={"A number, the dimension"},
		outputs={"A matrix"},
})

april_set_doc(matrixComplex.."get", {
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

april_set_doc(matrixComplex.."set", {
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
		outputs = { "The caller matrixComplex" }, })

april_set_doc(matrixComplex.."raw_get", {
		class = "method",
		summary = "Returns the value stored at a given RAW position.",
		params = {
		  "RAW position",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrixComplex.."raw_set", {
		class = "method",
		summary = "Sets the value of a given RAW position.",
		params = {
		  "RAW position", 
		},
		outputs = { "The caller matrixComplex" }, })

april_set_doc(matrixComplex.."offset", {
		class = "method",
		summary = "Returns the RAW offset position of matrixComplex data.",
		outputs = {
		  "A number with the RAW offset position",
		}, })

april_set_doc(matrixComplex.."fill", {
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

april_set_doc(matrixComplex.."set_use_cuda", {
		class = "method",
		summary = "Indicates if use or not CUDA for math operations.",
		params = {
		  "A boolean",
		},
		outputs = {
		  "The caller object (itself)",
		}, })

april_set_doc(matrixComplex.."dim", {
		class = "method",
		summary = "Returns a table with the size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc(matrixComplex.."dim", {
		class = "method",
		summary = "Returns the size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrixComplex.."stride", {
		class = "method",
		summary = "Returns a table with the stride size of each dimension.",
		outputs = {
		  "A table",
		}, })

april_set_doc(matrixComplex.."stride", {
		class = "method",
		summary = "Returns the stride size of a given dimension number.",
		params = {
		  "A number indicating the dimension, between 1 and num_dims",
		},
		outputs = {
		  "A number",
		}, })

april_set_doc(matrixComplex.."slice", {
		class = "method",
		summary = "Returns a sub-matrixComplex that is a slice of caller matrixComplex.",
		description = {
		  "Returns a sub-matrixComplex that is a slice of caller matrixComplex.",
		  "This method returns a sub-matrixComplex which references the",
		  "parent matrixComplex (not copy the data). Optionally it is possible",
		  "to do a deep copy (clone) of the data.",
		},
		params = {
		  "A table with the first position of the sub-matrixComplex",
		  "A table with the sizes of each dimension for the sub-matrixComplex",
		  { "A boolean indicating if do or not a clone [optional]. By",
		    "default it is set to false", },
		},
		outputs = {
		  "A matrixComplex object (sub-matrixComplex)",
		}, })

april_set_doc(matrixComplex.."rewrap", {
		class = "method",
		summary = "Reinterprets the data as with other dimensions.",
		description = {
		  "Returns a matrixComplex which references to the caller, but",
		  "reinterpreting the internal data with the given array of",
		  "dimension sizes.",
		  "The caller matrixComplex must be a contiguous chunk of data.",
		},
		params = {
		  { "A table with the size of each dimension.",
		    "The number of dimensions could be different of ",
		    "caller matrixComplex.", },
		},
		outputs = {
		  "A matrixComplex object (referencing the caller matrixComplex)",
		}, })

april_set_doc(matrixComplex.."select", {
		class = "method",
		summary = "Returns an slice result of select given dimension at given index.",
		description = {
		  "Returns an slice result of select given dimension at given index.",
		  "The matrixComplex has one less dimension because the selected dimension",
		  "is removed.",
		},
		params = {
		  { "A number with the selected dimension" },
		  { "A number with the selected index" },
		},
		outputs = {
		  "A matrixComplex object (referencing the caller matrixComplex)",
		}, })

april_set_doc(matrixComplex.."clone", {
		class = "method",
		summary = "Returns a deep copy (clone) of the caller matrixComplex.",
		outputs = {
		  "A matrixComplex object (cloned)",
		}, })

april_set_doc(matrixComplex.."transpose", {
		class = "method",
		summary = "Returns transposition of the caller matrixComplex.",
		description = {
		  "Returns transposition of the caller matrixComplex.",
		  "The returned matrixComplex references the original.",
		},
		outputs = {
		  "A matrixComplex object (transposed)",
		}, })

april_set_doc(matrixComplex.."diag", {
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

april_set_doc(matrixComplex.."toTable", {
		class = "method",
		summary = "Returns a Lua table with the data of the matrixComplex.",
		description = {
		  "Returns a Lua table with the data of the matrixComplex.",
		  "The table is a copy of the data in row_major order.",
		},
		outputs = {
		  "A Lua table",
		}, })

april_set_doc(matrixComplex.."add", {
		class = "method",
		summary = "Returns the addition of caller and other matrixComplex.",
		description = {
		  "Returns a new matrixComplex which is addition of caller and other matrixComplex.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrixComplex",
		},
		outputs = {
		  "A new matrixComplex result of addition",
		}, })

april_set_doc(matrixComplex.."sub", {
		class = "method",
		summary = "Returns the subtraction of caller and other matrixComplex.",
		description = {
		  "Returns a new matrixComplex which is the subtraction of caller and other matrixComplex.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrixComplex",
		},
		outputs = {
		  "A new matrixComplex result of subtraction",
		}, })

april_set_doc(matrixComplex.."mul", {
		class = "method",
		summary = "Returns the multiplication of caller and other matrixComplex.",
		description = {
		  "Returns a new matrixComplex which is the multiplication of caller and other matrixComplex.",
		  "This method works with vector-vector, dot product,",
		  "vector-matrixComplex, and matrixComplex-matrixComplex multiplication, depending",
		  "on the matrices dimensions.",
		  "It uses BLAS operations.",
		},
		params = {
		  "Another matrixComplex (2D matrixComplex or a vector)",
		},
		outputs = {
		  "A new matrixComplex result of multiplication",
		}, })

april_set_doc(matrixComplex.."cmul", {
		class = "method",
		summary = "Computes IN-PLACE the component-wise multiplication of caller and other matrix.",
		params = {
		  "Another matrix",
		},
		outputs = {
		  "The caller matrix",
		}, })

april_set_doc(matrixComplex.."axpy", {
		class = "method",
		summary = "BLAS AXPY operation IN-PLACE: Y = Y + alpha * X.",
		params = {
		  "Alpha constant",
		  "Another matrixComplex, X",
		},
		outputs = {
		  "The caller matrixComplex (a vector), Y (itself)",
		}, })

april_set_doc(matrixComplex.."gemv", {
		class = "method",
		summary = "BLAS GEMV operation IN-PLACE: Y = beta * Y + alpha * A * X.",
		params = {
		  ["alpha"] = "Alpha constant",
		  ["beta"]  = "Beta constant",
		  ["trans_A"]  = {
		    "A boolean indicating if transpose or not matrixComplex A.",
		    "It is [optional], by default is false",
		  },
		  ["A"] = "Another matrixComplex, A (a 2D matrixComplex)",
		  ["X"] = "Another matrixComplex, X (a vector)",
		},
		outputs = {
		  "The caller matrixComplex (a vector), Y (itself)",
		}, })

april_set_doc(matrixComplex.."gemm", {
		class = "method",
		summary = "BLAS GEMM operation IN-PLACE: C = beta * C + alpha * A * B.",
		params = {
		  ["alpha"] = "Alpha constant",
		  ["beta"]  = "Beta constant",
		  ["trans_A"]  = {
		    "A boolean indicating if transpose or not matrixComplex A.",
		    "It is [optional], by default is false",
		  },
		  ["trans_B"]  = {
		    "A boolean indicating if transpose or not matrixComplex B.",
		    "It is [optional], by default is false",
		  },
		  ["A"] = "Another matrixComplex, A (a 2D matrixComplex)",
		  ["B"] = "Another matrixComplex, B (a 2D matrixComplex)",
		},
		outputs = {
		  "The caller matrixComplex, C (itself)",
		}, })

april_set_doc(matrixComplex.."dot", {
		class = "method",
		summary = "BLAS DOT operation IN-PLACE: v = X  dot  Y",
		params = {
		  "Another matrixComplex, Y (a vector)",
		},
		outputs = {
		  "A number with the dot product",
		}, })

april_set_doc(matrixComplex.."scal", {
		class = "method",
		summary = "BLAS SCAL operation IN-PLACE.",
		params = {
		  "The scale factor",
		},
		outputs = {
		  "The caller matrixComplex (itself)."
		}, })

april_set_doc(matrixComplex.."norm2", {
		class = "method",
		summary = "BLAS NRM2 operation.",
		outputs = {
		  "A number with the norm-2 of caller matrixComplex."
		}, })

april_set_doc(matrixComplex.."copy",
	      {
		class = "method",
		summary = "Copy the values from another matrixComplex using BLAS",
		params  = { "A source matrixComplex" },
		outputs = { "The caller matrixComplex instance" },
})

april_set_doc(matrixComplex.."linear",
	      {
		class = "method",
		summary = "Initializes with linear integers",
		params  = { "First integer value [optional], by default 0",
			    "Step value [optional], by default 1", },
		outputs = { "The caller matrixComplex instance" },
})

april_set_doc(matrixComplex.."is_contiguous",
	      {
		class = "method",
		summary = "Returns true if the matrixComplex data is contiguous at memory",
		outputs = { "A boolean" },
})

april_set_doc(matrixComplex.."scalar_add",
	      {
		class = "method",
		summary = "Adds a scalar IN-PLACE",
		params  = { "A number" },
		outputs = { "The caller matrixComplex instance" },
})


april_set_doc(matrixComplex.."to_float",
	      {
		class = "method",
		summary = "Returns a matrix (with float resolution)",
		description = {
		  "Converts the given matrixComplex to a matrix (with float)",
		  "with the one additional dimension of size two, where the",
		  "real and imaginary part will be stored together.",
		  "The extra dimension will be the last.",
		},
		outputs = { "A matrix instance" },
})

april_set_doc(matrixComplex.."conj",
	      {
		class = "method",
		summary = "Applies the conjugate of all the elements IN-PLACE",
		outputs = { "The caller matrixComplex instance" },
})

april_set_doc(matrixComplex.."real",
	      {
		class = "method",
		summary = "Returns the real part of the caller matrix",
		outputs = { "A matrix instance" },
})

april_set_doc(matrixComplex.."img",
	      {
		class = "method",
		summary = "Returns the imaginary part of the caller matrix",
		outputs = { "A matrix instance" },
})

april_set_doc(matrixComplex.."abs",
	      {
		class = "method",
		summary = "Returns the abs of the polar representation",
		outputs = { "A matrix instance" },
})

april_set_doc(matrixComplex.."angle",
	      {
		class = "method",
		summary = "Returns the angle of the polar representation",
		outputs = { "A matrix instance" },
})

april_set_doc(matrixComplex.."sliding_window",
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
		outputs = { "An instance of matrixComplex.__sliding_window__" },
})

april_set_doc(matrixComplex.__sliding_window__,
	      {
		class       = "class",
		summary     = "Sliding window for matrixComplex objects",
})

april_set_doc(matrixComplex.__sliding_window__.."get_matrix",
	      {
		class       = "method",
		summary     = "Returns the matrixComplex generated by the window",
		params      = { {"A matrixComplex [optional], to be used instead",
				 "of alloc a new one"} },
		outputs     = { "A matrixComplex instance" },
})

april_set_doc(matrixComplex.__sliding_window__.."next",
	      {
		class       = "method",
		summary     = "Moves the window to the next position",
		outputs     = { "The caller sliding_window object" },
})

april_set_doc(matrixComplex.__sliding_window__.."is_end",
	      {
		class       = "method",
		summary     = "Returns true when it finishes",
		outputs     = { "A boolean" },
})

april_set_doc(matrixComplex.__sliding_window__.."iterate",
	      {
		class       = "method",
		summary     = "Returns an iterator function: for mat in s:iterate() do ... end",
		outputs     = { "An iterator function" },
})

matrix.__generic__.__make_generic_call__(matrixComplex)

matrixComplex.meta_instance.__newindex =
  matrix.__generic__.__make_generic_newindex__(matrixComplex)

matrix.__generic__.__make_generic_index__(matrixComplex)

matrixComplex.meta_instance.__tostring =
  matrix.__generic__.__make_generic_print__("MatrixComplex",
                                            function(value)
                                              return string.format("%26s",tostring(value))
  end)

