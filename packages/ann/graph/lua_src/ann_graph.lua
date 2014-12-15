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
  function(self, name)
    self.nodes = { input = { in_edges = {}, out_edges = {} } }
    self.name = name or "ann_graph"
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
    if tt_src ~= "table" then src = { src } end
    for i=1,#src do
      local v = src[i]
      assert(class.is_a(v, ann.components.base) or v == "input" ,
             "Needs an ann component or 'input' string as source")
      table.insert(self.nodes[v].out_edges, dst)
    end
    self.nodes[dst] = { in_edges = src, out_edges = {} }
    return self
  end

ann_graph_methods.build =
  april_doc{
    class = "method",
    summary = "Builds the ANN component",
    description = "See april_help(ann.components.base..'build')",
    params = { "An optional table" },
  } ..
  function(self, tbl)
    local tbl = tbl or {}
    assert(#self.nodes.input.out_edges > 0,
           "Connections from 'input' node are needed")
    assert(self.nodes.output, "Connections to 'output' node are needed")
    self.order = topological_sort(self.nodes, "input")
    assert(self.order[1] == "input" and self.order[#self.order] == "output")
    -- remove 'input' and 'output' strings from topological order table
    table.remove(self.order, 1)
    table.remove(self.order, #self.order)
    --
    local nodes = self.nodes
    local input_size = tbl.input or 0
    local weights = tbl.weights or {}
    local components = { [self.name] = self }
    local function compute_size(isize, tbl)
      local sz = 0
      for _,k in ipairs(tbl) do
        if k == "input" then sz = sz + isize
        else sz = sz + k:get_output_size() end
      end
      return sz
    end
    for _,obj in ipairs(self.order) do
      local node = nodes[obj]
      obj:build{ input = compute_size(input_size, node.in_edges),
                 weights = weights }
      obj:copy_components(components)
    end
    self.is_built = true
    self.input_size = compute_size(tbl.input or 0, nodes.input.out_edges)
    self.output_size = compute_size(tbl.output or 0, nodes.output.in_edges)
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
  local error_outputs_table = { output=input }
  local accumulate = function(dst, e)
    error_outputs_table[dst] = error_outputs_table[dst] or matrix.as(e)
    error_outputs_table[dst]:axpy(1.0, e)
  end
  for i=#self.order,1,-1 do
    local obj = self.order[i]
    local node = self.nodes[obj]
    local error_input = error_outputs_table[obj]
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
  self.error_output_token = error_outputs_table.input
  return self.error_output_token
end

ann_graph_methods.compute_gradients = function(self, weight_grads)
  local weight_grads = weight_grads or {}
  for _,obj in self.order do obj:compute_gradients(weight_grads) end
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
  return self.is_built or false
end

ann_graph_methods.debug_info = function(self)
  error("Not implemented")
end

ann_graph_methods.get_input_size = function(self)
  return self.input_size
end

ann_graph_methods.get_output_size = function(self)
  return self.output_size
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
    outputs_table[obj] = obj:precompute_output_size(input)
  end
  return compose(self.nodes.output.in_edges, outputs_table)
end

ann_graph_methods.clone = function(self)
  local graph = ann.graph(self.name)
  for obj,node in pairs(self.edges) do
    local new_obj = (obj ~= "input" and obj ~= "output" and obj:clone()) or obj
    for _,dst in ipairs(node.out_edges) do graph:connect(new_obj, dst) end
  end
  return graph
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

ann_graph_methods.copy_weights = function(self)
  local dict = {}
  for _,obj in ipairs(self.order) do obj:copy_weights(dict) end
  return dict
end

ann_graph_methods.copy_components = function(self)
  local dict = {}
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
