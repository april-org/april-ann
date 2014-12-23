local mop = matrix.op

-- AUXILIARY FUNCTIONS

local function forward_asserts(self)
  assert(self:get_is_built(),
         "Build method should be called before")
end

local function backprop_asserts(self)
  assert(self:get_is_built(),
         "Build method should be called before")
  assert(rawget(self,"output_token"),
         "Forward method should be called before")
end

local function compute_gradients_asserts(self)
  assert(self:get_is_built(),
         "Build method should be called before")
  assert(rawget(self,"output_token"),
         "Forward method should be called before")
  assert(rawget(self,"error_output_token"),
         "Backprop method should be called before")
end

local function forward_finish(self, input, output)
  self:set_input(input)
  self:set_output(output)
end

local function backprop_finish(self, input, output)
  self:set_error_input(input)
  self:set_error_output(output)
end

--

local function reverse(t,...)
  return iterator.range(#t,1,-1):map(function(k) return t[k] end):table(), ...
end

-- Returns a table with the reverse topological sort given a table of nodes and
-- the start object. Additionally, it is returned a boolean which indicates if
-- the network is recurrent.

-- TODO: check cicles
local function topological_sort(nodes, obj, visited, result)
  local result = result or {}
  local visited = visited or {}
  local node = nodes[obj]
  local recurrent = false
  visited[obj] = 'r'
  for _,dst in ipairs(node.out_edges) do
    if not visited[dst] then
      local _,r = topological_sort(nodes, dst, visited, result)
      recurrent = recurrent or r
    elseif visited[dst] == 'r' then
      recurrent = true
    end
  end  
  result[#result+1] = obj
  visited[obj] = recurrent and 'R' or 'b'
  return result,recurrent,visited
end

-- Composes a tokens.vector.bunch given a table with multiple objects and
-- a dictionary from these objects to tokens. In case #tbl == 1, instead of
-- a tokens.vector.bunch instance, the value dict[tbl[1]] would be returned.
local function compose(tbl, dict)
  local result
  if #tbl > 1 then
    result = tokens.vector.bunch()
    for i = 1,#tbl do
      result:push_back( assert(dict[tbl[i]]) )
    end
  else
    result = dict[tbl[1]]
  end
  return result
end

local function ann_graph_topsort(self)
  self.order,self.recurrent,self.colors =
    reverse( topological_sort(self.nodes, "input") )
  assert(self.order[1] == "input" and self.order[#self.order] == "output")
  -- remove 'input' and 'output' strings from topological order table
  table.remove(self.order, 1)
  table.remove(self.order, #self.order)
end

------------------------------------------------------------------------------

ann = ann or {}

local ann_graph_methods
ann.graph,ann_graph_methods = class("ann.graph", ann.components.lua)

april_set_doc(ann.graph, {
                class = "class",
                summary = "An ANN component for flow graphs", })

ann.graph.constructor =
  april_doc{
    class = "method",
    summary = "constructor",
    params = { "A name string" },
  } ..
  function(self, name, components, connections)
    ann.components.lua.constructor(self, name)
    self.nodes = { input = { in_edges = {}, out_edges = {} } }
    if components and connections then
      for src,dst in iterator(connections):map(table.unpack) do
        if components[dst] ~= "input" then
          local src = iterator(src):map(function(i) return components[i] end):table()
          self:connect(src, components[dst])
        end
      end
    end
    -- for truncated BPTT
    self.bptt_step = 0 -- controls current step number
    self.backstep  = 1 -- indicates truncation length (it can be math.huge)
  end

ann_graph_methods.connect =
  april_doc{
    class = "method",
    summary = "Performs the connection between two ANN components",
    description = {
      "The connections are described in a many-to-one way, so multiple",
      "source components can be defined as input of one component.",
      "If multiple components are defined, a tokens.vector.bunch instance",
      "would be received as input of the destination component.",
      "Additionally, 'input' string can be used as source to indicate the",
      "graph input token. Similarly, 'output' string can be used as destination",
      "to produce the graph output token.",
    },
    params = {
      "An ANN component, a table of multiple ANN components. 'input' string is fine.",
      "An ANN component or 'output' string.",
    },
  } ..
  function(self, src, dst)
    assert(class.is_a(dst, ann.components.base) or dst == "output",
           "Needs an ann component or 'output' string as destination")
    assert(not self.nodes[dst], "Overwriting a previously defined destination")
    local tt_src = type(src)
    -- just to ensure src as a table
    if tt_src ~= "table" then src = { src } end
    -- traverse every input and take note of dst in out_edges of every input
    for i=1,#src do
      local v = src[i]
      assert(class.is_a(v, ann.components.base) or v == "input" ,
             "Needs an ann component or 'input' string as source")
      table.insert(self.nodes[v].out_edges, dst)
    end
    -- take note of the given input table as input edges of the given dst
    self.nodes[dst] = { in_edges = src, out_edges = {} }
    return self
  end

ann_graph_methods.build = function(self, tbl)
  local tbl = tbl or {}
  assert(#self.nodes.input.out_edges > 0,
         "Connections from 'input' node are needed")
  assert(self.nodes.output, "Connections to 'output' node are needed")
  -- build the topological sort, which will be stored at self.order
  ann_graph_topsort(self)
  --
  local nodes = self.nodes
  local input_size = tbl.input or 0
  local weights = tbl.weights or {}
  local components = { [self.name] = self }
  -- computes the sum of the elements of sizes which belong to tbl
  local function sum_sizes(tbl, sizes)
    return iterator(tbl):map(function(obj) return sizes[obj] end):reduce(math.add(), 0)
  end
  --
  local input_sizes = {}
  local output_sizes = { input = input_size }
  -- precompute input/output sizes
  for _,obj in ipairs(self.order) do
    input_sizes[obj] = obj:get_input_size()
    output_sizes[obj] = obj:get_output_size()
  end
  input_sizes.output = sum_sizes(nodes.output.in_edges, output_sizes)
  --
  for _,obj in ipairs(self.order) do
    local node = nodes[obj]
    april_assert(#node.out_edges > 0,
                 "Node %s doesn't have output connections",
                 obj:get_name())
    local _,_,aux = obj:build{ input = sum_sizes(node.in_edges, output_sizes),
                               output = sum_sizes(node.out_edges, input_sizes),
                               weights = weights }
    for k,v in pairs(aux) do assert(not components[k]) components[k] = v end
    input_sizes[obj]  = obj:get_input_size()
    output_sizes[obj] = obj:get_output_size()
  end
  for k=2,#nodes.input.out_edges do
    assert(input_sizes[nodes.input.out_edges[1]] == input_sizes[nodes.input.out_edges[k]],
           "All input connection should have the same input size")
  end
  (ann.components.lua.."build")(self,
                                { weights = weights,
                                  input = input_sizes[nodes.input.out_edges[1]],
                                  output = sum_sizes(nodes.output.in_edges, output_sizes) })
  return self,weights,components
end

-- traverse the graph following the topological order (self.order), composing
-- the input of every component by using previous component outputs, and storing
-- the output of every component at the table outputs_table
ann_graph_methods.forward = function(self, input, during_training)
  forward_asserts(self)
  local outputs_table = { input=input }
  for _,obj in ipairs(self.order) do
    local node = self.nodes[obj]
    local input = compose(node.in_edges, outputs_table)
    outputs_table[obj] = obj:forward(input, during_training)
  end
  local output = compose(self.nodes.output.in_edges, outputs_table)
  forward_finish(self, input, output)
  return output
end

-- traverse the graph following inverse topological order (self.order),
-- composing the error input of every component by accumulating previous
-- component error outputs, and storing the error output of every component at
-- the table error_outputs_table
ann_graph_methods.backprop = function(self, input)
  backprop_asserts(self)
  local error_inputs_table = { output=input }
  local accumulate = function(dst, e)
    if e ~= tokens.null() then
      local err = error_inputs_table[dst]
      if not err then error_inputs_table[dst] = e:clone()
      else err = err:axpy(1.0, e) end
    end
  end
  for _,obj in ipairs(self.nodes.output.in_edges) do accumulate(obj, input) end
  for i=#self.order,1,-1 do
    local obj = self.order[i]
    local node = self.nodes[obj]
    local error_input = error_inputs_table[obj]
    local error_output = obj:backprop(error_input)
    if class.is_a(error_output, tokens.vector.bunch) then
      assert(error_output:size() == #node.in_edges)
      for j,e in error_output:iterate() do
        accumulate(node.in_edges[j], e)
      end
    else
      accumulate(node.in_edges[1], error_output)
    end
  end
  backprop_finish(self, input, error_inputs_table.input)
  return error_inputs_table.input
end

ann_graph_methods.compute_gradients = function(self, weight_grads)
  compute_gradients_asserts(self)
  local weight_grads = weight_grads or {}
  for _,obj in ipairs(self.order) do obj:compute_gradients(weight_grads) end
  return weight_grads
end

ann_graph_methods.reset = function(self, n)
  for _,obj in self.order do obj:reset() end
  (ann.components.lua.."reset")(self, n)
  self.bptt_step = 0
end

ann_graph_methods.precompute_output_size = function(self, tbl)
  local function sum(t,other)
    for i=1,#t do t[i] = t[i] + other[i] end
    return t
  end
  local function compose(t,dict)
    iterator(t):reduce(function(acc,n) return sum(acc, dict[n]) end, {})
  end
  --
  local outputs_table = { input=tbl }
  for _,obj in ipairs(self.order) do
    local node = self.nodes[obj]
    local input = compose(node.in_edges, outputs_table)
    outputs_table[obj] = obj:precompute_output_size(input, #node.in_edges)
  end
  return compose(self.nodes.output.in_edges, outputs_table)
end

ann_graph_methods.clone = function(self)
  local graph = ann.graph(self.name)
  graph.nodes = util.clone(self.nodes)
  return graph
end

ann_graph_methods.to_lua_string = function(self, format)
  local cnns = {}
  if not rawget(self,"order") then ann_graph_topsort(self) end
  local ext_order = iterator(self.order):table()
  ext_order[#ext_order+1] = "input"
  ext_order[#ext_order+1] = "output"
  local ext_obj2id = table.invert(ext_order)
  for id,dst in ipairs(ext_order) do
    cnns[id] = {
      iterator(ipairs(self.nodes[dst].in_edges)):
      map(function(j,src) return j,ext_obj2id[src] end):table(),
      id,
    }
  end
  local str = {
    "ann.graph(", "%q"%{self.name} , ",",
    util.to_lua_string(ext_order, format), ",",
    util.to_lua_string(cnns, format), ")",
  }
  return table.concat(str)
end

ann_graph_methods.set_use_cuda = function(self, v)
  for k,v in pairs(self.nodes) do
    if type(k) ~= "string" then
      v:set_use_cuda(v)
    end
  end
  (ann.components.lua.."set_use_cuda")(self, v)
end

ann_graph_methods.copy_weights = function(self, dict)
  local dict = dict or {}
  for _,obj in ipairs(self.order) do obj:copy_weights(dict) end
  (ann.components.lua.."copy_weights")(self, dict)
  return dict
end

ann_graph_methods.copy_components = function(self, dict)
  local dict = dict or {}
  for _,obj in ipairs(self.order) do obj:copy_components(dict) end
  (ann.components.lua.."copy_components")(self, dict)
  return dict
end

ann_graph_methods.get_component = function(self, name)
  if self.name == name then return self end
  for _,obj in ipairs(self.order) do
    local c = obj:get_component(name)
    if c then return c end
  end
end

ann_graph_methods.get_is_recurrent = function(self)
  return rawget(self,recurrent)
end

ann_graph_methods.set_bptt_truncation = function(self, backstep)
  self.backstep = (backstep <= 0) and math.huge or backstep
end

---------------------------------------------------------------------------

local bind_methods
ann.graph.bind,bind_methods = class("ann.graph.bind", ann.components.lua)

ann.graph.bind.constructor = function(self, name)
  ann.components.lua.constructor(self, name)
end

bind_methods.build = function(self, tbl)
  local _,w,c = (ann.components.lua.."build")(self, tbl)
  if self:get_input_size() == 0 then
    self.input_size = self:get_output_size()
  end
  if self:get_output_size() == 0 then
    self.output_size = self:get_input_size()
  end
  assert(self:get_input_size() == self:get_output_size(),
         "Unable to compute input/output sizes")
  return self,w,c
end

bind_methods.forward = function(self, input, during_training)
  forward_asserts(self)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  local output = matrix.join(2, iterator(input:iterate()):
                               map(function(i,m)
                                   assert(#m:dim() == 2,
                                          "Needs flattened input matrices")
                                   return i,m
                               end):table())
  forward_finish(self, input, output)
  return output
end

bind_methods.backprop = function(self, input)
  backprop_asserts(self)
  local output = tokens.vector.bunch()
  local pos = 1
  for _,m in self:get_input():iterate() do
    local dest = pos + m:dim(2) - 1
    local slice = input(':', {pos, dest})
    pos = dest + 1
    output:push_back(slice)
  end
  backprop_finish(self, input, output)
  return output
end

bind_methods.precompute_output_size = function(self, tbl)
  assert(#tbl == 1, "Needs a flattened input")
  return tbl
end

---------------------------------------------------------------------------

local add_methods
ann.graph.add,add_methods = class("ann.graph.add", ann.components.lua)

ann.graph.add.constructor = function(self, name)
  ann.components.lua.constructor(self, name)
end

add_methods.build = function(self, tbl)
  local _,w,c = (ann.components.lua.."build")(self, tbl)
  if rawget(self,"input_size") and rawget(self,"output_size") then
    assert( (self.input_size % self.output_size) == 0,
      "Output size should be a multiple of input size")
  end
  return self,w,c
end

add_methods.forward = function(self, input, during_training)
  forward_asserts(self)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  local output = input:at(1):clone()
  for i=2,input:size() do output:axpy(1.0, input:at(i)) end
  forward_finish(self, input, output)
  return output
end

add_methods.backprop = function(self, input)
  backprop_asserts(self)
  local output = tokens.vector.bunch()
  for i=1,self:get_input():size() do
    output:push_back(input)
  end
  backprop_finish(self, input, output)
  return output
end

add_methods.precompute_output_size = function(self, tbl, n)
  assert(#tbl == 1, "Needs a flattened input")
  return iterator(tbl):map(math.div(nil,n)):table()
end

---------------------------------------------------------------------------

local index_methods
ann.graph.index,index_methods = class("ann.graph.index", ann.components.lua)

ann.graph.index.constructor = function(self, n, name)
  self.n = assert(n, "Needs a number as first argument")
  ann.components.lua.constructor(self, name)
end

index_methods.forward = function(self, input, during_training)
  forward_asserts(self)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  local output = input:at(self.n)
  forward_finish(self, input, output)
  return output
end

index_methods.backprop = function(self, input)
  backprop_asserts(self)
  local output = tokens.vector.bunch()
  for i = 1, self.n-1 do output:push_back(tokens.null()) end
  output:push_back(input)
  for i = self.n+1, self:get_input():size() do output:push_back(tokens.null()) end
  backprop_finish(self, input, output)
  return output
end

index_methods.clone = function(self)
  return ann.graph.index(self.n, self.name)
end

index_methods.to_lua_string = function(self, format)
  return "ann.graph.index(%d,%q)" % {self.n, self.name}
end

---------------------------------------------------------------------------

local cmul_methods
ann.graph.cmul,cmul_methods = class("ann.graph.cmul", ann.components.lua)

ann.graph.cmul.constructor = function(self, name)
  self.name = name or ann.generate_name()
end

cmul_methods.build = function(self, tbl)
  local _,w,c = (ann.components.lua.."build")(self, tbl)
  if rawget(self,"input_size") and rawget(self,"output_size") then
    assert( self.input_size == 2*self.output_size,
      "Output size should be a multiple of input size")
  end
  return self,w,c
end

cmul_methods.forward = function(self, input, during_training)
  forward_asserts(self)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  assert(input:size() == 2, "Needs a tokens.vector.bunch with two components")
  local output = mop.cmul(input:at(1), input:at(2))
  forward_finish(self, input, output)
  return output
end

cmul_methods.backprop = function(self, input)
  backprop_asserts(self)
  local i = self:get_input()
  local output = tokens.vector.bunch()
  output:push_back( mop.cmul(i:at(2), input) )
  output:push_back( mop.cmul(i:at(1), input) )
  backprop_finish(self, input, output)
  return output
end

cmul_methods.precompute_output_size = function(self, tbl, n)
  assert(#tbl == 1, "Needs a flattened input")
  return iterator(tbl):map(math.div(nil,2)):table()
end

---------------------------------------------------------------------------

ann.graph.test = function()
  local nodes = {
    a = { out_edges = { 'b', 'c' }, in_edges = { } },
    b = { out_edges = { 'c', 'd' }, in_edges = { 'a' } },
    c = { out_edges = { 'd' }, in_edges = { 'a', 'b' } },
    d = { out_edges = { }, in_edges = { 'b', 'c' } },
  }
  local result,recurrent = reverse( topological_sort(nodes, 'a') )
  utest.check.TRUE( iterator.zip(iterator(result),
                                 iterator{ 'a', 'b', 'c', 'd' }):
                    reduce(function(acc,a,b) return acc and a==b end, true) )
  utest.check.FALSE(recurrent)
  --
  local nodes = {
    a = { out_edges = { 'b' } },
    b = { out_edges = { 'c' } },
    c = { out_edges = { 'd', 'e' } },
    d = { out_edges = { 'b', 'f' } },
    e = { out_edges = { 'f' } },
    f = { out_edges = { } },
  }
  local result,recurrent,colors = reverse( topological_sort(nodes, 'a') )
  utest.check.TRUE( iterator.zip(iterator(result),
                                 iterator{ 'a', 'b', 'c', 'e', 'd', 'f' }):
                    reduce(function(acc,a,b) return acc and a==b end, true) )
  utest.check.TRUE(recurrent)
  local ref_colors = { a='R', b='R', c='R', d='R', e='b', f='b' }
  utest.check.TRUE(
    iterator(pairs(colors)):
    reduce(function(acc,k,v) return acc and ref_colors[k]==v end, true)
  )
  --
  local a = matrix(3,4):linear()
  local b = matrix(3,5):linear()
  local c = matrix(3,10):linear()
  local tbl = { 'c', 'b', 'a' }
  local dict = { a = a, b = b, c = c }
  local result = compose(tbl, dict)
  utest.check.TRUE( result:at(1) == c )
  utest.check.TRUE( result:at(2) == b )
  utest.check.TRUE( result:at(3) == a )
end
