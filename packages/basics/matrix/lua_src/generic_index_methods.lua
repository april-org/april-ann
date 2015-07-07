matrix = matrix or {}
matrix.__generic__ = matrix.__generic__ or {}

local index_function =
  april_doc{
    class = "method",
    summary = {
      "Returns new allocated matrix filtered by the given dim and",
      "index matrix parameters.",
    },
    params = {
      "A dimension number",
      { "A table, matrixBool or matrixInt32 indicating which",
        "indices will be taken." },
    },
    outputs = {
      { "A new allocated matrix instance, or nil if 2nd argument",
        "has zero selected components", },
    },
  } ..
  function(self,dim,idx)
    assert(type(dim) == "number",
           "Needs a number as first argument")
    if type(idx) == "table" then idx = matrixInt32(idx)
    elseif class.is_a(idx, matrixBool) then idx = idx:to_index()
    end
    if not idx then return nil end
    assert(class.is_a(idx, matrixInt32),
           "Needs a matrixInt32 second argument (index)")
    local idx = idx:squeeze()
    assert(#idx:dim() == 1, "Needs a rank 1 tensor as second argument (index)")
    local d = self:dim()
    assert(dim >= 1 and dim <= #d, "Dimension argument out-of-bounds")
    local dim_bound = d[dim]
    d[dim] = idx:size()
    local ctor   = class.of(self)
    local result = ctor(table.unpack(d))
    d[dim] = 1
    local self_sw = self:sliding_window{ size=d, step=d }
    local dest_sw   = result:sliding_window{ size=d, step=d }
    local result_submat,self_submat
    idx:map(function(p)
        april_assert(p >= 1 and p <= dim_bound,
                     "Index number %d out-of-bounds", p)
        assert(not dest_sw:is_end())
        self_sw:set_at_window(p)
        result_submat = dest_sw:get_matrix(result_submat)      
        self_submat = self_sw:get_matrix(self_submat)
        result_submat:copy(self_submat)
        dest_sw:next()
    end)
    return result
  end

local index_fill_function =
  april_doc{
    class = "method",
    summary = {
      "Fills the indexed dim,index components of the caller matrix.",
    },
    params = {
      "A dimension number",
      { "A table, matrixBool or matrixInt32 indicating which",
        "indices will be taken." },
      "A number value for filling.",
    },
    outputs = {
      "The caller matrix.",
    },
  } ..
  function(self,dim,idx,val)
    assert(type(dim) == "number",
           "Needs a number as first argument")
    if type(idx) == "table" then idx = matrixInt32(idx)
    elseif class.is_a(idx, matrixBool) then idx = idx:to_index()
    end
    if not idx then return self end
    assert(class.is_a(idx, matrixInt32),
           "Needs a matrixInt32 second argument (index)")
    local idx = idx:squeeze()
    assert(#idx:dim() == 1,
           "Needs a rank 1 tensor as second argument (index)")
    local d = self:dim()
    assert(dim >= 1 and dim <= #d,"Dimension argument out-of-bounds")
    local dim_bound = d[dim]
    d[dim] = 1
    local sw = self:sliding_window{ size=d, step=d }
    local mat
    idx:map(function(p)
        april_assert(p >= 1 and p <= dim_bound,
                     "Index number %d out-of-bounds", i)
        sw:set_at_window(p)
        mat = sw:get_matrix(mat)
        mat:fill(val)
    end)
    return self
  end

local index_copy_function =
  april_doc{
    class = "method",
    summary = {
      "Copies a matrix into the indexed dim,index components of the caller matrix.",
    },
    params = {
      "A dimension number",
      { "A table, matrixBool or matrixInt32 indicating which",
        "indices will be taken." },
      "A matrix with data to be copied.",
    },
    outputs = {
      "The caller matrix.",
    },
  } ..
  function(self,dim,idx,other)
    assert(type(dim) == "number",
           "Needs a number as first argument")
    if type(idx) == "table" then idx = matrixInt32(idx)
    elseif class.is_a(idx, matrixBool) then idx = idx:to_index()
    end
    if not idx then return self end
    assert(class.is_a(idx, matrixInt32),
           "Needs a matrixInt32 second argument (index)")
    assert(class.of(self) == class.of(other),
           "Self and other must be same matrix type")
    local idx = idx:squeeze()
    assert(#idx:dim() == 1,
           "Needs a rank 1 tensor as second argument (index)")
    local d = self:dim()
    assert(dim >= 1 and dim <= #d,"Dimension argument out-of-bounds")
    local dim_bound = d[dim]
    d[dim] = 1
    local other_sw = other:sliding_window{ size=d, step=d }
    local self_sw = self:sliding_window{ size=d, step=d }
    local other_submat,self_submat
    idx:map(function(p)
        april_assert(p >= 1 and p <= dim_bound,
                     "Index number %d out-of-bounds", i)
        assert(not other_sw:is_end())
        self_sw:set_at_window(p)
        other_submat = other_sw:get_matrix(other_submat)
        self_submat = self_sw:get_matrix(self_submat)
        self_submat:copy(other_submat)
        other_sw:next()
    end)
    return self
  end  

function matrix.__generic__.__make_index_methods__(ctor)
  class.extend(ctor, "index",        index_function)
  class.extend(ctor, "indexed_fill", index_fill_function)
  class.extend(ctor, "indexed_copy", index_copy_function)
end