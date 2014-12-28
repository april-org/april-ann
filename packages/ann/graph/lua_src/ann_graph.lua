local mop        = matrix.op
local null_token = tokens.null()

-- The ann.graph package allows to implement ANNs as if they were a graph of
-- components. The nodes of the graph are operations (ANN components), and the
-- edges are connections between output and input of the components. The graph
-- is described as a dictionary of nodes, indexed by the object itself (the Lua
-- reference, not the object name). For every node it is stored a table with:
-- in_edges, out_edges, in_delay_values, out_delay_values.

-- AUXILIARY FUNCTIONS

-- Receives token which can be: tokens.vector.bunch, token.matrix,
-- token.sparse_matrix. Returns the bunch size (mini-batch size) of the given
-- token (that is, the size of the first dimension).
local get_bunch_size = function(tk)
  if class.is_a(tk, tokens.vector.bunch) then
    tk = tk:at(1)
  end
  return tk:dim(1)
end

-- Given an ANN component or a string, returns the obj:get_name() or the string
-- itself.
local name_of = function(obj)
  if type(obj) == "string" then return obj end
  return obj:get_name()
end

-- Function hepers to traverse node in/out edges list, they return an iterator.
local node_fwd_in_edges_it, node_all_in_edges_it
local node_fwd_out_edges_it, node_all_out_edges_it

do
  
  -- Returns an iterator of three values: number, object, delay value.
  local function generic_iterator(edges, delay_values)
    return iterator.zip(iterator(ipairs(edges)), iterator(delay_values))
  end
  
  -- Iterator which traverses only in forward connections.
  node_fwd_in_edges_it = function(node)
    return generic_iterator(node.in_edges, node.in_delay_values):
    filter(function(k,v,d) return d == 0 end)
  end

  -- Iterator which traverses only out forward connections.
  node_fwd_out_edges_it = function(node)
    return generic_iterator(node.out_edges, node.out_delay_values):
    filter(function(k,v,d) return d == 0 end)
  end
  
  -- Iterator which traverses all in connections.
  node_all_in_edges_it = function(node)
    return generic_iterator(node.in_edges, node.in_delay_values)
  end
  
  -- Iterator which traverses all out connections.
  node_all_out_edges_it = function(node)
    return generic_iterator(node.out_edges, node.out_delay_values)
  end
  
end

-- Returns a node table with empty lists of edges and delay values.
local function node_constructor()
  return { in_edges = {}, out_edges = {},
           in_delay_values = {}, out_delay_values = {} }
end

-- Connects two objects using the given table of nodes to store the data.
local function node_connect(nodes, src, dst, delay)
  table.insert(nodes[src].out_edges, dst)
  table.insert(nodes[src].out_delay_values, delay)
  table.insert(nodes[dst].in_edges, src)
  table.insert(nodes[dst].in_delay_values, delay)
end

----------------------------------------------------------------------------

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

------------------------------------------------------------------------------

-- Reverses the table given as first argument. The rest of arguments are
-- returned as they are given.
local function reverse(t, ...)
  return iterator.range(#t,1,-1):map(function(k) return t[k] end):table(), ...
end

-- Returns a table with the reverse topological sort given a table of nodes and
-- the start object. The topological sort is computed following only the
-- non-delayed connections.
local function topological_sort(nodes, obj, visited, result, back_nodes)
  local visited = visited or {}
  local result = result or {}
  local back_nodes = back_nodes or {}
  local node = nodes[obj]
  local recurrent = false
  visited[obj] = 'r'
  for _,dst in node_fwd_out_edges_it(node) do
    if not visited[dst] then
      local _,r = topological_sort(nodes, dst, visited, result, back_nodes)
      recurrent = recurrent or r
    elseif visited[dst] == 'r' then
      -- loop check
      recurrent = true
      back_nodes[obj] = true
    end
  end  
  result[#result+1] = obj
  visited[obj] = recurrent and 'R' or 'b'
  return result,recurrent,visited,back_nodes
end

-- Composes a tokens.vector.bunch given an iterator with multiple objects and a
-- function to map object,delay pairs to tokens. In case #iterator == 1, instead
-- of a tokens.vector.bunch instance, the value dict(tbl[1]) would be returned.
local function compose(it, dict)
  local result,n=0
  for k,obj,delay in it do
    if k == 2 then
      local aux = result
      result = tokens.vector.bunch()
      result:push_back(aux)
    end
    if k > 1 then
      result:push_back( assert(dict(obj, delay)) )
    else
      result = assert(dict(obj, delay))
    end
  end
  return result
end

-- Computes the topological sort of an ANN graph object. Only recurrent delayed
-- connections are possible. All delayed connections must be backward
-- connections, because topological sort uses only non-delayed connections.
local function ann_graph_topsort(self)
  local recurrent,colors,back_nodes
  self.order, recurrent, colors, back_nodes =
    reverse( topological_sort(self.nodes, "input") )
  for obj,_ in pairs(self.nodes) do
    assert( colors[obj], "Unable to compute topological sort. Check your "..
              "delayed connections, they should be backward edges.")
  end
  assert(not recurrent, "Unable to sort ANN with 0-delay recurrent connections")
  assert(self.order[1] == "input")
  -- remove 'input' and 'output' strings from topological order table
  table.remove(self.order, 1)
  for i,v in ipairs(self.order) do
    if v=='output' then table.remove(self.order, i) break end
  end
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
  function(self, name, components, connections, backstep)
    ann.components.lua.constructor(self, { name=name })
    -- for truncated BPTT
    self.max_delay = 0  -- maximum delay in the network
    self.bptt_step = 0  -- controls current BPTT step number
    self.bptt_data = {} -- array with ANN state for every BPTT step
    self:set_bptt_truncation(backstep or 0) -- indicates truncation length
    --
    self.nodes = { input = node_constructor() }
    if components and connections then
      for sources,delays,dst in iterator(connections):map(table.unpack) do
        if components[dst] ~= "input" then
          for k,v in ipairs(sources) do
            local delay  = delays[k]
            local source = components[v]
            self:connect(source, components[dst], delay)
          end
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
      "to produce the graph output token. Multiple calls will be aggregated.",
    },
    params = {
      "An ANN component, a table of multiple ANN components. 'input' string is fine.",
      "An ANN component or 'output' string.",
      "The delay value, by default it is 0 [optional]",
    },
    outputs = {
      "A function which can be called to concatenate connections in a forward way",
    },
  } ..
  function(self, src, dst, delay)
    delay = delay or 0
    assert(delay >= 0, "Delay must be a >= 0")
    assert(class.is_a(dst, ann.components.base) or dst == "output",
           "Needs an ann component or 'output' string as destination")
    local tt_src = type(src)
    -- just to ensure src as a table
    if tt_src ~= "table" then src = { src } end
    --
    local function check(v)
      self.nodes[v] = self.nodes[v] or node_constructor()
      return self.nodes[v]
    end
    check(dst)
    local node = self.nodes[dst]
    -- traverse src and take note of the connections
    for i=1,#src do
      local v = src[i]
      assert(class.is_a(v, ann.components.base) or v == "input" ,
             "Needs an ann component or 'input' string as source")
      check(v)
      node_connect(self.nodes, v, dst, delay)
      self.max_delay = math.max(self.max_delay, delay)
    end
    return function(dst2, delay) return self:connect(dst, dst2, delay) end
  end

ann_graph_methods.delayed =
  april_doc{
    class = "method",
    summary = "Performs a delayed connection between two ANN components",
    description = {
      "This method is a wrapper over connect, but using a dealy of 1 by default",
    },
    params = {
      "An ANN component, a table of multiple ANN components. 'input' string is fine.",
      "An ANN component or 'output' string.",
      "The delay value, by default it is 1 [optional]",
    },
    outputs = {
      "A function which can be called to concatenate connections in a forward way",
    },
  } ..
function(self, src, dst, delay) assert(not delay or delay > 0) return self:connect(src, dst, delay or 1) end

-- ann_graph_methods.remove =
--   april_doc{
--     class = "method",
--     summary = "Removes one node and all its connections",
--     params = {
--       "An ANN component.",
--     },
--     outputs = {
--       "The caller object.",
--     },
--   } ..
--   function(self, obj)
--     self.is_built = false
--     local node = assert(self.nodes[obj], "Unable to locate the given component")
--     self.nodes[obj] = nil
--     for dst in iterator(node.out_edges) do
--       self.nodes[dst].in_edges = iterator(self.nodes[dst].in_edges):
--       filter(function(v) return v~=obj end):table()
--     end
--     for src in iterator(node.in_edges) do
--       self.nodes[src].out_edges = iterator(self.nodes[src].out_edges):
--       filter(function(v) return v~=obj end):table()
--     end
--     return self
--   end

-- ann_graph_methods.replace =
--   april_doc{
--     class = "method",
--     summary = "Replaces one node by another",
--     params = {
--       "The ANN component to be replaced.",
--       "The new ANN component.",
--     },
--     outputs = {
--       "The caller object.",
--     },
--   } ..
--   function(self, old, new)
--     self.is_built = false
--     local node = assert(self.nodes[old], "Unable to locate the given component")
--     self:remove(old)
--     for dst in iterator(node.out_edges) do
--       self:connect(new, dst)
--     end
--     for src in iterator(node.in_edges) do
--       self:connect(src, new)
--     end
--     return self
--   end

ann_graph_methods.dot_graph =
  function(self, filename)
    local f = io.open(filename, "w")
    f:write("digraph %s {\nrankdir=BT;\n"%{self:get_name()})
    for tbl in iterator{self.order,{'input','output'}} do
      for obj in iterator(tbl) do
        local node = self.nodes[obj]
        local src = (obj ~= "input" and obj ~= "output" and obj:get_name()) or obj
        local shape, edge_style = "ellipse", "solid"
        if obj ~= "input" and obj ~= "output" then
          f:write('%s [label="%s (%s)",shape="%s"];\n'%{ src, src, type(obj), shape })
        end
        for k,obj2,delay in node_all_out_edges_it(node) do
          local edge_style = (delay == 0) and edge_style or "dashed"
          local constraint = (delay == 0) and "true" or "false"
          local dst = (obj2 ~= "input" and obj2 ~= "output" and obj2:get_name()) or obj2
          f:write('%s -> %s [label="%s",style=%s,constraint=%s];\n' %
                    { src, dst, delay, edge_style, constraint })
        end
      end
    end
    f:write("input [shape=none];\n")
    f:write("output [shape=none];\n")
    f:write("{rank=source; input;}\n")
    f:write("{rank=sink; output;}\n")
    f:write("}\n")
    f:close()
  end

ann_graph_methods.build = function(self, tbl)
  assert(self.backstep >= self.max_delay,
         "Impossible to build the network with the given BPTT truncation parameter")
  local tbl = tbl or {}
  assert(#self.nodes.input.out_edges > 0,
         "Connections from 'input' node are needed")
  assert(self.nodes.output, "Connections to 'output' node are needed")
  -- build the topological sort, which will be stored at self.order
  ann_graph_topsort(self)
  self.recurrent = (self.max_delay > 0)
  if self.recurrent then
    if self.backstep == 0 then self:set_bptt_truncation(math.huge) end
  end
  --
  local nodes = self.nodes
  local input_size = tbl.input or 0
  local output_size = tbl.output or 0
  local weights = tbl.weights or {}
  local components = { [self.name] = self }
  -- computes the sum of the elements of sizes which belong to tbl
  local function sum_sizes(tbl, sizes)
    local zero = false
    local sz = iterator(tbl):map(function(obj) zero=(sizes[obj] == 0) return sizes[obj] end):reduce(math.add(), 0)
    if zero then sz = 0 end
    return sz
  end
  local function check_sizes(tbl, sizes)
    local sz
    for i=1,#sizes do sz = sz or sizes[i] assert(sz == sizes[i]) end
    return sz
  end
  --
  local input_sizes = { input = input_size, output = output_size }
  local output_sizes = { input = input_size, output = output_size }
  -- precompute input/output sizes
  for _,obj in ipairs(self.order) do
    input_sizes[obj]  = obj:get_input_size()
    output_sizes[obj] = obj:get_output_size()
  end
  if input_sizes.output == 0 then
    output_size = sum_sizes(nodes.output.in_edges, output_sizes)
    input_sizes.output = output_size
    output_sizes.output = output_size
  end
  --
  for _,obj in ipairs(self.order) do
    local node = nodes[obj]
    april_assert(#node.out_edges > 0,
                 "Node %s doesn't have output connections",
                 obj:get_name())
    local _,_,aux = obj:build{ input = sum_sizes(node.in_edges, output_sizes),
                               output = check_sizes(node.out_edges, input_sizes),
                               weights = weights }
    for k,v in pairs(aux) do components[k] = v end
    input_sizes[obj]  = obj:get_input_size()
    output_sizes[obj] = obj:get_output_size()
  end
  -- FIXME: problem with components where input size is undefined
  -- for k=2,#nodes.input.out_edges do
  --   assert(input_sizes[nodes.input.out_edges[1]] == input_sizes[nodes.input.out_edges[k]],
  --          "All input connection should have the same input size")
  -- end
  (ann.components.lua.."build")(self,
                                { weights = weights,
                                  input = input_sizes[nodes.input.out_edges[1]],
                                  output = sum_sizes(nodes.output.in_edges, output_sizes) })
  return self,weights,components
end

local empty_table = {}
-- Traverse the graph following the topological order (self.order), composing
-- the input of every component by using previous component outputs. The state
-- of every component is stored at table bptt_data, indexed by time iteration
-- and object name. This states are needed to implement BPTT algorithm.
ann_graph_methods.forward = function(self, input, during_training)
  self.gradients_computed = false
  forward_asserts(self)
  local bunch_size = get_bunch_size(input)
  ------------------
  -- BPTT section --
  ------------------
  local bptt = self.bptt_data
  if self:get_is_recurrent() then
    self.bptt_step = self.bptt_step + 1
    if self.bptt_step > self.backstep then self.bptt_step = 1 end
  else
    self.bptt_step = 1
  end
  local backstep  = self.backstep
  local bptt_step = self.bptt_step
  -- current time iteration is initialized with default values for input object
  bptt[bptt_step] = { input = { input = input, output = input } }
  -- This function receives an object and the delay, and returns the output
  -- activation of the object at the corresponding time iteration. In case of
  -- nil activation, a default matrix with zeros will be returned.
  local function retrieve_output(obj, delay)
    local pos = bptt_step - delay
    if backstep < math.huge then pos = (pos - 1) % backstep + 1 end
    local value = ( (bptt[pos] or empty_table)[name_of(obj)] or empty_table).output
    assert(delay > 0 or value) -- sanity check for non-delayed connections
    local sz = (obj == "input" and self:get_output_size()) or obj:get_output_size()
    april_assert(sz > 0,
                 "Unable to initialize or retrieve default activation of component %s (%s)",
                 name_of(obj), type(obj))
    return value or matrix(bunch_size, sz):zeros()
  end
  ------------------
  for _,obj in ipairs(self.order) do
    local node   = self.nodes[obj]
    local input  = compose(node_all_in_edges_it(node), retrieve_output)
    local output = obj:forward(input, during_training)
    -- copy state of ALL compounds of obj, it can be a complex object
    bptt[bptt_step] = obj:copy_state(bptt[bptt_step])
  end
  local node   = self.nodes.output
  local output = compose(node_all_in_edges_it(node), retrieve_output)
  forward_finish(self, input, output)
  ------------------
  -- BPTT section --
  ------------------
  bptt[bptt_step].output = { input=output, output=output }
  ------------------
  return output
end

-- FIXME: fix the case when a component returns a tokens.vector.bunch instance,
-- it cannot be accumulated in the way is done.

-- Auxiliary function which accumulates the given error output at the inputs of
-- the given node data.
local accumulate_error_output = function(node, error_output, i, retrieve_state)
  -- refactored function
  local acc = function(dst, e, retrieve_state, ...)
    if e and e ~= null_token then
      local state = retrieve_state(dst, ...)
      local err = state.error_input
      if not err or err == null_token then state.error_input = e:clone()
      else err = err:axpy(1.0, e) end
    end
  end
  -- iterator over all in edges
  local edges_it = node_all_in_edges_it(node)
  if class.is_a(error_output, tokens.vector.bunch) then
    assert(error_output:size() == #node.in_edges)
    for j,e,_,src,delay in iterator.zip(iterator(error_output:iterate()),
                                        edges_it) do
      acc(src, e, retrieve_state, i, delay)
    end
  else
    local _,src,delay = edges_it:head()
    acc(src, error_output, retrieve_state, i, delay)
  end
end

-- Computes the gradients of all components in an ANN graph. Default zero
-- gradients are introduced in case of a component which has not input deltas
-- to compute the gradients.
local function ann_graph_compute_gradients(self)
  compute_gradients_asserts(self)
  local weight_grads = rawget(self,"grads") or {}
  for _,obj in ipairs(self.order) do
    if obj:get_error_input() and obj:get_error_input() ~= null_token then
      obj:compute_gradients(weight_grads)
    else
      for wname,w in pairs(obj:copy_weights()) do
        weight_grads[wname] = weight_grads[wname] or matrix.as(w):zeros()
      end
    end
  end
  self.grads = weight_grads
  self.gradients_computed = true
end

-- Applies the backprop method to all the components of an ANN graph. This
-- backprop function follows BPTT algorithm, so errors are back-propagated
-- trough space and time.
local function ann_graph_backprop(self)
  assert(self.bptt_step == 1 or self:get_is_recurrent(),
         "Unable to use BPTT in non recurrent networks")
  backprop_asserts(self)
  local bptt     = self.bptt_data
  local backstep = self.backstep
  -- BPTT SECTION --
  local function retrieve_state(obj, bptt_step, delay)
    -- local pos   = (backstep + bptt_step - delay - 1) % self.backstep + 1
    local pos = bptt_step - delay
    if backstep ~= math.huge then pos = (pos - 1) % backstep + 1 end
    bptt[pos] = bptt[pos] or {}
    local name = name_of(obj)
    bptt[pos][name] = bptt[pos][name] or {}
    return bptt[pos][name]
  end
  ------------------
  -- loop over time
  for i=self.bptt_step,1,-1 do
    local input = bptt[i].output.error_output
    -- accumulate error deltas of nodes connected to the output
    accumulate_error_output(self.nodes.output, input, i, retrieve_state)
    -- loop over components (loop over space)
    for j=#self.order,1,-1 do
      local obj = self.order[j]
      local node = self.nodes[obj]
      -- set the state of obj at time iteration i
      obj:set_state(bptt[i])
      local error_input = bptt[i][name_of(obj)].error_input
      if error_input and error_input ~= null_token then
        local error_output = obj:backprop(error_input)
        -- accumulate error output into all input connections
        accumulate_error_output(node, error_output, i, retrieve_state)
      end -- if error_input
    end -- for j in reverse topological order
    backprop_finish(self, input, bptt[i].input.error_input)
    -- compute and accumulate gradients of current iteration
    ann_graph_compute_gradients(self)
    -- BPTT SECTION --
    bptt[i].input.error_output = bptt[i].input.error_input or null_token
    ------------------
  end -- for i in reverse time
  return bptt[1].input.error_output
end

-- traverse the graph following inverse topological order (self.order),
-- composing the error input of every component by accumulating previous
-- component error outputs, and storing the error output of every component at
-- the table error_outputs_table
ann_graph_methods.backprop = function(self, input)
  -- keep the backprop input for a future use
  local bptt = assert(self.bptt_data[self.bptt_step], "Execute forward before")
  bptt.output.error_input  = input
  bptt.output.error_output = input
  if self.bptt_step == self.backstep or not self:get_is_recurrent() then
    return ann_graph_backprop(self)
  else
    return null_token
  end
end

-- Computes the gradients, forcing to execute backprop in case it is needed.
ann_graph_methods.compute_gradients = function(self, weight_grads)
  if not rawget(self,"gradients_computed") then
    -- ann_graph_backprop implements gradient computation
    ann_graph_backprop(self)
    assert(rawget(self,"gradients_computed"))
  end
  local weight_grads = weight_grads or {}
  for k,v in pairs(self.grads) do weight_grads[k] = v end
  return weight_grads
end

ann_graph_methods.copy_state = function(self, tbl)
  tbl = tbl or {}
  (ann.components.lua.."copy_state")(self, tbl)
  for _,obj in ipairs(self.order) do obj:copy_state(tbl) end
  -- TODO: compute input and output components state
  return tbl
end

ann_graph_methods.set_state = function(self, tbl)
  (ann.components.lua.."set_state")(self, tbl)
  for _,obj in ipairs(self.order) do obj:set_state(tbl) end
  -- TODO: set input and output components state
  return self
end

-- Sets to zero all the intermediate data, even the BPTT stuff.
ann_graph_methods.reset = function(self, n)
  for _,obj in ipairs(self.order) do obj:reset() end
  (ann.components.lua.."reset")(self, n)
  self.bptt_step = 0
  self.bptt_data = {}
  self.grads = {}
end

-- TODO: implement method precompute_output_size
ann_graph_methods.precompute_output_size = function(self, tbl)
  error("Not implemented")
  -- local function sum(t,other)
  --   for i=1,#t do t[i] = t[i] + other[i] end
  --   return t
  -- end
  -- local function compose(t,dict)
  --   iterator(t):reduce(function(acc,n) return sum(acc, dict[n]) end, {})
  -- end
  -- --
  -- local outputs_table = { input=tbl }
  -- for _,obj in ipairs(self.order) do
  --   local node = self.nodes[obj]
  --   local input = compose(node.in_edges, outputs_table)
  --   outputs_table[obj] = obj:precompute_output_size(input, #node.in_edges)
  -- end
  -- return compose(self.nodes.output.in_edges, outputs_table)
end

ann_graph_methods.clone = function(self)
  -- After cloning, the BPTT is truncated, so, it is recommended to avoid
  -- cloning when learning a sequence, it is better to clone after any
  -- sequence learning.
  local graph = ann.graph(self.name)
  graph.nodes = util.clone(self.nodes)
  return graph
end

ann_graph_methods.to_lua_string = function(self, format)
  -- After saving, the BPTT is truncated, so, it is recommended to avoid
  -- saving when learning a sequence, it is better to clone after any
  -- sequence learning.
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
      iterator(ipairs(self.nodes[dst].in_delay_values)):table(),
      id,
    }
  end
  local str = {
    "ann.graph(", "%q"%{self.name} , ",",
    util.to_lua_string(ext_order, format), ",",
    util.to_lua_string(cnns, format), ",",
    tostring(self.backstep - 1), ")",
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
  return rawget(self,"recurrent")
end

ann_graph_methods.set_bptt_truncation = function(self, backstep)
  self.backstep = backstep + 1
  assert(self.backstep > self.max_delay,
         "Unable to set the given BPTT truncation")
end

---------------------------------------------------------------------------

local bind_methods
ann.graph.bind,bind_methods = class("ann.graph.bind", ann.components.lua)

ann.graph.bind.constructor = function(self, tbl)
  if tbl and tbl.size then
    tbl.input, tbl.output, tbl.size = tbl.size, tbl.size, nil
  end
  ann.components.lua.constructor(self, tbl)
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

ann.graph.add.constructor = function(self, tbl)
  ann.components.lua.constructor(self, tbl)
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
  local i,output = 1
  while not input:at(i) or input:at(i) == null_token do i=i+1 end
  output = input:at(i):clone()
  for i=i+1,input:size() do
    local tk = input:at(i)
    if tk and tk ~= null_token then output:axpy(1.0, input:at(i)) end
  end
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
  ann.components.lua.constructor(self, { name=name })
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
  for i = 1, self.n-1 do output:push_back(null_token) end
  output:push_back(input)
  for i = self.n+1, self:get_input():size() do output:push_back(null_token) end
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

ann.graph.cmul.constructor = function(self, tbl)
  ann.components.lua.constructor(self, tbl)
end

cmul_methods.build = function(self, tbl)
  local _,w,c = (ann.components.lua.."build")(self, tbl)
  if self:get_output_size() == 0 then
    self.output_size = self:get_input_size() / 2
  end
  if self:get_input_size() == 0 then
    self.input_size = self:get_output_size() * 2
  end
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
  local a, b = input:at(1), input:at(2)
  local output
  if not a or a == null_token then
    output = b
  elseif not b or b == null_token then
    output = a
  else
    output = mop.cmul(a, b)
  end
  forward_finish(self, input, output)
  return output
end

cmul_methods.backprop = function(self, input)
  backprop_asserts(self)
  local i = self:get_input()
  local output = tokens.vector.bunch()
  local a, b = i:at(1), i:at(2)
  if b and b ~= null_token then
    output:push_back( mop.cmul(b, input) )
  else
    output:push_back( input )
  end
  if a and a ~= null_token then
    output:push_back( mop.cmul(a, input) )
  else
    output:push_back( input )
  end
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
    a = { out_edges = { 'b', 'c' }, out_delay_values = { 0, 0 } },
    b = { out_edges = { 'c', 'd' }, out_delay_values = { 0, 0 } },
    c = { out_edges = { 'd' }, out_delay_values = { 0 } },
    d = { out_edges = { } },
  }
  local result,recurrent = reverse( topological_sort(nodes, 'a') )
  utest.check.TRUE( iterator.zip(iterator(result),
                                 iterator{ 'a', 'b', 'c', 'd' }):
                    reduce(function(acc,a,b) return acc and a==b end, true) )
  utest.check.FALSE(recurrent)
  --
  local nodes = {
    a = { out_edges = { 'b' }, out_delay_values = { 0 } },
    b = { out_edges = { 'c' }, out_delay_values = { 0 } },
    c = { out_edges = { 'd', 'e' }, out_delay_values = { 0, 0 } },
    d = { out_edges = { 'b', 'f' }, out_delay_values = { 0, 0 } },
    e = { out_edges = { 'f' }, out_delay_values = { 0 } },
    f = { out_edges = { }, out_delay_values = { } },
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
end
