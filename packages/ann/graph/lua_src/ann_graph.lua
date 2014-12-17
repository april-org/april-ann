-- AUXILIARY FUNCTIONS

-- Returns a table with a topological sort given a table of nodes and the
-- start object.

-- TODO: check cicles
local function topological_sort(nodes, obj)
  local result = { }
  local queue = { obj }
  local in_counts = { }
  local i = 0
  while i < #queue do
    i = i + 1
    local current = queue[i]
    local node = nodes[current]
    result[#result+1] = current
    for _,dst in ipairs(node.out_edges) do
      in_counts[dst] = (in_counts[dst] or 0) + 1
      if in_counts[dst] == #nodes[dst].in_edges then
        queue[#queue+1] = dst
      end
    end
  end
  return result
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
  self.order = topological_sort(self.nodes, "input")
  assert(self.order[1] == "input" and self.order[#self.order] == "output")
  -- remove 'input' and 'output' strings from topological order table
  table.remove(self.order, 1)
  table.remove(self.order, #self.order)
end

------------------------------------------------------------------------------

ann = ann or {}

local ann_graph_methods
ann.graph,ann_graph_methods = class("ann.graph", ann.components.base)

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
    self.nodes = { input = { in_edges = {}, out_edges = {} } }
    self.name = name or ann.generate_name()
    if components and connections then
      for src,dst in iterator(connections):map(table.unpack) do
        if components[dst] ~= "input" then
          local src = iterator(src):map(function(i) return components[i] end):table()
          self:connect(src, components[dst])
        end
      end
    end
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

ann_graph_methods.build =
  april_doc{
    class = "method",
    summary = "Builds the ANN component",
    description = "See april_help(ann.components.base..'build')",
    params = { "An optional table with 'input', 'output' and 'weights' fields" },
  } ..
  function(self, tbl)
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
      local _,_,aux = obj:build{ input = sum_sizes(node.in_edges, output_sizes),
                                 output = sum_sizes(node.out_edges, input_sizes),
                                 weights = weights }
      for k,v in pairs(aux) do assert(not components[k]) components[k] = v end
      input_sizes[obj]  = obj:get_input_size()
      output_sizes[obj] = obj:get_output_size()
    end
    self.input_size  = sum_sizes(nodes.input.out_edges, input_sizes)
    self.output_size = sum_sizes(nodes.output.in_edges, output_sizes)
    self.is_built    = true
    return self,weights,components
  end

-- traverse the graph following the topological order (self.order), composing
-- the input of every component by using previous component outputs, and storing
-- the output of every component at the table outputs_table
ann_graph_methods.forward = function(self, input, during_training)
  local outputs_table = { input=input }
  for _,obj in ipairs(self.order) do
    local node = self.nodes[obj]
    local input = compose(node.in_edges, outputs_table)
    outputs_table[obj] = obj:forward(input, during_training)
  end
  self.input_token = input
  self.output_token = compose(self.nodes.output.in_edges, outputs_table)
  return self.output_token
end

-- traverse the graph following inverse topological order (self.order),
-- composing the error input of every component by accumulating previous
-- component error outputs, and storing the error output of every component at
-- the table error_outputs_table
ann_graph_methods.backprop = function(self, input)
  local error_inputs_table = { output=input }
  local accumulate = function(dst, e)
    local err = error_inputs_table[dst]
    if not err then error_inputs_table[dst] = e:clone()
    else err = err:axpy(1.0, e) end
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
  self.error_input_token = input
  self.error_output_token = error_inputs_table.input
  return self.error_output_token
end

ann_graph_methods.compute_gradients = function(self, weight_grads)
  local weight_grads = weight_grads or {}
  for _,obj in ipairs(self.order) do obj:compute_gradients(weight_grads) end
  return weight_grads
end

ann_graph_methods.reset = function(self, input)
  for _,obj in self.order do obj:reset() end
  self.input_token = nil
  self.output_token = nil
  self.error_input_token = nil
  self.error_output_token = nil
end

ann_graph_methods.get_name = function(self)
  return self.name
end

ann_graph_methods.get_weights_name = function(self)
  return nil
end

ann_graph_methods.has_weights_name = function(self)
  return false
end

ann_graph_methods.get_is_built = function(self)
  return self.order and true or false
end

ann_graph_methods.debug_info = function(self)
  error("Not implemented")
end

ann_graph_methods.get_input_size = function(self)
  return self.input_size or 0
end

ann_graph_methods.get_output_size = function(self)
  return self.output_size or 0
end

ann_graph_methods.get_input = function(self)
  return self.input_token
end

ann_graph_methods.get_output = function(self)
  return self.output_token
end

ann_graph_methods.get_error_input = function(self)
  return self.error_input_token
end

ann_graph_methods.get_error_output = function(self)
  return self.error_output_token
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
  if not rawget(self,order) then ann_graph_topsort(self) end
  local order = iterator(self.order):table()
  order[#order+1] = "input"
  order[#order+1] = "output"
  local obj2id = table.invert(order)
  for id,dst in ipairs(order) do
    cnns[id] = {
      iterator(ipairs(self.nodes[dst].in_edges)):
      map(function(j,src) return j,obj2id[src] end):table(),
      id,
    }
  end
  local str = {
    "ann.graph(", "%q"%{self.name} , ",",
    util.to_lua_string(order, format), ",",
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
  self.use_cuda = v
end

ann_graph_methods.get_use_cuda = function(self)
  return self.use_cuda or false
end

ann_graph_methods.copy_weights = function(self, dict)
  local dict = dict or {}
  for _,obj in ipairs(self.order) do obj:copy_weights(dict) end
  return dict
end

ann_graph_methods.copy_components = function(self, dict)
  local dict = dict or {}
  for _,obj in ipairs(self.order) do obj:copy_components(dict) end
  return dict
end

ann_graph_methods.get_component = function(self, name)
  if self.name == name then return self end
  for _,obj in ipairs(self.order) do
    local c = obj:get_component(name)
    if c then return c end
  end
end

---------------------------------------------------------------------------

local bind_methods
ann.graph.bind,bind_methods = class("ann.graph.bind", ann.components.base)

ann.graph.bind.constructor = function(self, name)
  self.name = name or ann.generate_name()
end

bind_methods.build = function(self, tbl)
  local tbl = tbl or {}
  if tbl.input and tbl.input ~= 0 then
    self.size = tbl.input
    assert(not tbl.output or tbl.output == 0 or tbl.output == tbl.input)
  end
  if tbl.output ~= 0 then
    self.size = tbl.output
    assert(not tbl.input or tbl.input == 0 or tbl.input == tbl.output)
  end
  if not self.size then self.size = 0 end
  return self,tbl.weights or {},{ [self.name] = self }
end

bind_methods.forward = function(self, input, during_training)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  local output = matrix.join(2, iterator(input:iterate()):
                               map(function(i,m)
                                   assert(#m:dim() == 2,
                                          "Needs flattened input matrices")
                                   return i,m
                               end):table())
  self.input_token = input
  self.output_token = output
  return output
end

bind_methods.backprop = function(self, input)
  local output = tokens.vector.bunch()
  local pos = 1
  for _,m in self.input_token:iterate() do
    local dest = pos + m:dim(2) - 1
    local slice = input(':', {pos, dest})
    pos = dest + 1
    output:push_back(slice)
  end
  self.error_input_token = input
  self.error_output_token = output
  return output
end

bind_methods.compute_gradients = function(self, weight_grads)
  return weight_grads or {}
end

bind_methods.reset = function(self, input)
  self.input_token = nil
  self.output_token = nil
  self.error_input_token = nil
  self.error_output_token = nil
end

bind_methods.get_name = function(self)
  return self.name
end

bind_methods.get_weights_name = function(self)
  return nil
end

bind_methods.has_weights_name = function(self)
  return false
end

bind_methods.get_is_built = function(self)
  return (self.size ~= nil)
end

bind_methods.debug_info = function(self)
  error("Not implemented")
end

bind_methods.get_input_size = function(self)
  return self.size or 0
end

bind_methods.get_output_size = function(self)
  return self.size or 0
end

bind_methods.get_input = function(self)
  return self.input_token
end

bind_methods.get_output = function(self)
  return self.output_token
end

bind_methods.get_error_input = function(self)
  return self.error_input_token
end

bind_methods.get_error_output = function(self)
  return self.error_output_token
end

bind_methods.precompute_output_size = function(self, tbl)
  assert(#tbl == 1, "Needs a flattened input")
  return tbl
end

bind_methods.clone = function(self)
  return ann.graph.bind(self.name)
end

bind_methods.to_lua_string = function(self, format)
  local str = { "ann.graph.bind(", "%q"%{self.name}, ")" }
  return table.concat(str)
end

bind_methods.set_use_cuda = function(self, v)
  self.use_cuda = v
end

bind_methods.get_use_cuda = function(self)
  return self.use_cuda or false
end

bind_methods.copy_weights = function(self, dict)
  return dict or {}
end

bind_methods.copy_components = function(self, dict)
  return dict or {}
end

bind_methods.get_component = function(self, name)
  if self.name == name then return self end
end

---------------------------------------------------------------------------

local sum_methods
ann.graph.sum,sum_methods = class("ann.graph.sum", ann.components.base)

ann.graph.sum.constructor = function(self, name)
  self.name = name or ann.generate_name()
end

sum_methods.build = function(self, tbl)
  local tbl = tbl or {}
  if tbl.input and tbl.input ~= 0 then
    self.input_size = tbl.input
  else
    self.input_size = 0
  end
  if tbl.output ~= 0 then
    self.output_size = tbl.output
  else
    self.output_size = 0
  end
  if self.input_size ~= 0 and self.output_size ~= 0 then
    assert( (self.input_size % self.output_size) == 0,
      "Input size mod output size has to be zero")
  end
  return self, tbl.weights or {}, { [self.name] = self }
end

sum_methods.forward = function(self, input, during_training)
  assert(class.is_a(input, tokens.vector.bunch),
         "Needs a tokens.vector.bunch as input")
  local output = input:at(1):clone()
  for i=2,input:size() do output:axpy(1.0, input:at(i)) end
  self.input_token = input
  self.output_token = output
  return output
end

sum_methods.backprop = function(self, input)
  local output = tokens.vector.bunch()
  for i=1,self.input_token:size() do
    output:push_back(input)
  end
  self.error_input_token = input
  self.error_output_token = output
  return output
end

sum_methods.compute_gradients = function(self, weight_grads)
  return weight_grads or {}
end

sum_methods.reset = function(self, input)
  self.input_token = nil
  self.output_token = nil
  self.error_input_token = nil
  self.error_output_token = nil
end

sum_methods.get_name = function(self)
  return self.name
end

sum_methods.get_weights_name = function(self)
  return nil
end

sum_methods.has_weights_name = function(self)
  return false
end

sum_methods.get_is_built = function(self)
  return (self.input_size ~= nil)
end

sum_methods.debug_info = function(self)
  error("Not implemented")
end

sum_methods.get_input_size = function(self)
  return self.input_size or 0
end

sum_methods.get_output_size = function(self)
  return self.output_size or 0
end

sum_methods.get_input = function(self)
  return self.input_token
end

sum_methods.get_output = function(self)
  return self.output_token
end

sum_methods.get_error_input = function(self)
  return self.error_input_token
end

sum_methods.get_error_output = function(self)
  return self.error_output_token
end

sum_methods.precompute_output_size = function(self, tbl, n)
  assert(#tbl == 1, "Needs a flattened input")
  return iterator(tbl):map(math.div(nil,n)):table()
end

sum_methods.clone = function(self)
  return ann.graph.sum(self.name)
end

sum_methods.to_lua_string = function(self, format)
  local str = { "ann.graph.sum(", "%q"%{self.name}, ")" }
  return table.concat(str)
end

sum_methods.set_use_cuda = function(self, v)
  self.use_cuda = v
end

sum_methods.get_use_cuda = function(self)
  return self.use_cuda or false
end

sum_methods.copy_weights = function(self, dict)
  return dict or {}
end

sum_methods.copy_components = function(self, dict)
  return dict or {}
end

sum_methods.get_component = function(self, name)
  if self.name == name then return self end
end

---------------------------------------------------------------------------
ann.graph.test = function()
  local nodes = {
    a = { out_edges = { "b", "c" }, in_edges = { } },
    b = { out_edges = { "c", "d" }, in_edges = { "a" } },
    c = { out_edges = { "d" }, in_edges = { "a", "b" } },
    d = { out_edges = { }, in_edges = { "b", "c" } },
  }
  local result = topological_sort(nodes, "a")
  utest.check.TRUE( iterator.zip(iterator(result), iterator{ "a", "b", "c", "d" }):
                    reduce(function(acc,a,b) return acc and a==b end, true) )
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
