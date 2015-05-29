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
  -- search the first token which is not a tokens.vector.bunch
  while class.is_a(tk, tokens.vector.bunch) do
    local j,tk2 = 1,nil
    while not tk2 or tk2 == null_token do tk2 = tk:at(j) j=j+1 end
    tk = tk2
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
-- A node has three properties:
--   - out_edges: a Lua table with a list of nodes
--   - out_delay_values: a Lua table with the delay of nodes in previous list
--   - in_edges: idem as out_edges
--   - in_edges_delay_values: idem out_delay_values
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
  -- assert(rawget(self,"error_output_token"),
  -- "Backprop method should be called before")
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
  local result = tokens.vector.bunch()
  for k,obj,delay in it do result:push_back( assert(dict(obj, delay)) ) end
  local result = (result:size() == 1) and result:at(1) or result
  return result
end

-- Computes the topological sort of an ANN graph object calling the function
-- topological_sort (defined above). Only recurrent delayed connections are
-- possible. Forward delayed connections can be used, and the function calls
-- iteratively to topological_sort to compute the order of **forward delayed**
-- nodes. A node is considered **forward delayed** when all its inputs are
-- delayed.
local function ann_graph_topsort(self)
  local input_name  = self.input_name
  local output_name = self.output_name
  local recurrent,visited,back_nodes
  -- compute topological order by using only non delayed connections, starting
  -- at input node
  self.order, recurrent, visited, back_nodes =
    reverse( topological_sort(self.nodes, input_name) )
  assert(not recurrent, "Unable to sort ANN with 0-delay recurrent connections")
  -- recompute topological order using non delayed connections from other nodes
  -- which only have delayed input connections (**forward delayed** nodes)
  for obj,node in pairs(self.nodes) do
    if not visited[obj] then
      assert(node_fwd_in_edges_it(node):size() == 0) -- sanity check
      local order,recurrent = reverse( topological_sort(self.nodes, obj,
                                                        visited, { }, back_nodes) )
      assert(not recurrent, "Unable to sort ANN with 0-delay recurrent connections")
      -- Because this nodes only have delayed input connections, they should be
      -- first in the topological order, so current self.order table is appended
      -- into order table...
      for i=1,#self.order do table.insert(order, self.order[i]) end
      -- and use the new order table as self.order
      self.order = order
    end
  end
  for obj,_ in pairs(self.nodes) do
    april_assert( visited[obj],
                  "Unable to compute topological sort, there are unreachable nodes: %s (%s)",
                  tostring(obj), name_of(obj) )
  end
  -- remove input_name and output_name strings from topological order table
  for i,v in ipairs(self.order) do
    if v==output_name or v==input_name then table.remove(self.order, i) end
  end
end

------------------------------------------------------------------------------

ann = ann or {}

local ann_graph_methods
ann.graph,ann_graph_methods = class("ann.graph", ann.components.lua)

april_set_doc(ann.graph, {
                class = "class",
                summary = "An ANN component based in graphs of other components", })

ann.graph.constructor =
  april_doc{
    class = "method",
    summary = "constructor",
    params = { "A name string" },
  } ..
  function(self, name, components, connections, backstep)
    ann.components.lua.constructor(self, { name=name })
    -- for truncated BPTT
    self.max_delay   = 0  -- maximum delay in the network
    self.bptt_step   = 0  -- controls current BPTT step number
    -- A with ANN state for every BPTT step, this array can be shared
    -- between other ann.graph instances in a composed graph. The sharing is
    -- done at reset and build method.
    self.bptt_data   = {}
    self.backstep    = backstep or math.huge -- indicates truncation length
    self.input_name  = self:get_name() .. "::input"
    self.output_name = self:get_name() .. "::output"
    --
    self.nodes = { }
    if components and connections then
      for sources,delays,dst in iterator(connections):map(table.unpack) do
        if components[dst] ~= self.input_name then
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
      "This function allows variadic arguments, allowing to stack multiple",
      "connections in one call.",
    },
    params = {
      "An ANN component, a table of multiple ANN components. 'input' string is fine.",
      "An ANN component",
      "...",
      "An ANN component or 'output' string.",
      "The delay value, by default it is 0 [optional]",
    },
    outputs = {
      "The last component in the path",
    },
  } ..
  function(self, src, ...)
    local args        = table.pack(...)
    local delay       = (type(args[#args]) ~= "number") and 0 or table.remove(args, #args)
    local input_name  = self.input_name
    local output_name = self.output_name
    local function name_mangling(v)
      return (v=="input" and input_name) or(v=="output" and output_name) or v
    end
    --
    self.is_built = false
    assert(delay >= 0, "Delay must be a >= 0")
    local tt_src = type(src)
    -- just to ensure src as a table
    if tt_src ~= "table" then src = { src } end
    --
    local function check(v)
      self.nodes[v] = self.nodes[v] or node_constructor()
      return self.nodes[v]
    end
    for _,dst in ipairs(args) do
      local dst = name_mangling(dst)
      assert(class.is_a(dst, ann.components.base) or dst == output_name,
             "Needs an ann component or 'output' string as destination")
      check(dst)
      local node = self.nodes[dst]
      -- traverse src and take note of the connections
      for i=1,#src do
        local v = name_mangling(src[i])
        assert(class.is_a(v, ann.components.base) or v == input_name ,
               "Needs an ann component or 'input' string as source")
        check(v)
        node_connect(self.nodes, v, dst, delay)
        self.max_delay = math.max(self.max_delay, delay)
      end
      src = { dst }
    end
    return args[#args]
  end

ann_graph_methods.delayed =
  april_doc{
    class = "method",
    summary = "Performs a delayed connection between two ANN components",
    description = {
      "This method is a wrapper over connect, but using a dealy of 1 by default.",
      "Additionally, this method doesn't support multiple connections.",
    },
    params = {
      "An ANN component, a table of multiple ANN components. 'input' string is fine.",
      "An ANN component or 'output' string.",
      "The delay value, by default it is 1 [optional]",
    },
    outputs = {
      "The last connected component",
    },
  } ..
function(self, src, dst, delay)
    assert(not delay or (type(delay) == "number" and delay > 0),
           "Incorrect arguments")
    return self:connect(src, dst, delay or 1)
end

ann_graph_methods.get_input_name = function(self)
  return self.input_name
end

ann_graph_methods.get_output_name = function(self)
  return self.output_name
end

-- for debug
ann_graph_methods.show_nodes = function(self,level)
  level = level or 0
  if level == 0 then print("# (level) name type") end
  for obj,_ in pairs(self.nodes) do
    local is_child = class.is_a(obj, ann.graph)
    printf("%s(%d) %-30s %s%s\n",
           iterator.duplicate(" "):take(level*2):concat(""),
           level, name_of(obj), type(obj),
           is_child and " +" or "")
    if is_child then obj:show_nodes(level+1) end
  end
end

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

ann_graph_methods.dot_graph = april_doc{
  class="method",
  summary="Writes a graphviz dot file, for subjective interpretation of the graph",
  params={ "The output filename" },
}..
  function(self, filename)
    assert(self:get_is_built(), "Needs a built component")
    local input_name = self.input_name
    local output_name = self.output_name
    local f = io.open(filename, "w")
    f:write("digraph %s {\nrankdir=BT;\n"%{self:get_name()})
    for tbl in iterator{self.order,{input_name,output_name}} do
      for obj in iterator(tbl) do
        local node = self.nodes[obj]
        local dst = name_of(obj)
        local shape, edge_style = "ellipse", "solid"
        if obj ~= input_name and obj ~= output_name then
          f:write('%q [label="%s (%s)",shape="%s"];\n'%{ dst, dst, type(obj), shape })
        end
        for k,obj2,delay in node_all_in_edges_it(node) do
          local edge_style = (delay == 0) and edge_style or "dashed"
          local constraint = (delay == 0) and "true" or "false"
          local src = name_of(obj2)
          f:write('%q -> %q [headlabel=<<font color="red">[%d]</font>>,label="%d",style=%s,constraint=%s];\n' %
                    { src, dst, k, delay, edge_style, constraint })
        end
      end
    end
    f:write("%q [shape=none];\n"%{input_name})
    f:write("%q [shape=none];\n"%{output_name})
    f:write("{rank=source; %q;}\n"%{input_name})
    f:write("{rank=sink; %q;}\n"%{output_name})
    f:write("}\n")
    f:close()
  end

ann_graph_methods.build = function(self, tbl, bptt_data)
  assert(self.backstep >= self.max_delay,
         "Impossible to build the network with the given BPTT truncation parameter")
  local tbl = tbl or {}
  assert(#self.nodes[self.input_name].out_edges > 0,
         "Connections from 'input' node are needed")
  assert(self.nodes[self.output_name], "Connections to 'output' node are needed")
  -- build the topological sort, which will be stored at self.order
  if not self:get_is_built() then ann_graph_topsort(self) end
  self.recurrent = (self.max_delay > 0)
  -- change bptt_data by the given one
  if bptt_data then
    self.bptt_data = bptt_data
  else
    bptt_data = self.bptt_data
  end
  --
  local input_name  = self.input_name
  local output_name = self.output_name
  local nodes       = self.nodes
  local input_size  = tbl.input or 0
  local output_size = tbl.output or 0
  local weights     = tbl.weights or {}
  local components  = { [self.name] = self }
  -- computes the sum of the elements of sizes which belong to tbl
  local function sum_sizes(tbl, sizes)
    local zero = false
    local sz = 0
    for obj in iterator(tbl) do
      if sizes[obj] == 0 then zero=true break end
      sz = sz + sizes[obj]
    end
    if zero then sz = 0 end
    return sz
  end
  local function check_sizes(tbl, sizes)
    local sz
    for i=1,#sizes do sz = sz or sizes[i] assert(sz == sizes[i]) end
    return sz
  end
  --
  local input_sizes = { [input_name] = input_size, [output_name] = output_size }
  local output_sizes = { [input_name] = input_size, [output_name] = output_size }
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
    local bptt_data = bptt_data
    if not class.is_a(obj, ann.graph) then bptt_data = nil end
    local _,aux
    if bptt_data then
      _,_,aux = obj:build({ input = sum_sizes(node.in_edges, output_sizes),
                            output = check_sizes(node.out_edges, input_sizes),
                            weights = weights }, bptt_data)
      if obj:get_is_recurrent() then self.recurrent = true end
    else
      _,_,aux = obj:build({ input = sum_sizes(node.in_edges, output_sizes),
                            output = check_sizes(node.out_edges, input_sizes),
                            weights = weights })
    end
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
                                  input = input_sizes[nodes[input_name].out_edges[1]],
                                  output = sum_sizes(nodes[output_name].in_edges, output_sizes) })
  return self,weights,components
end

local empty_table = {}
-- Traverse the graph following the topological order (self.order), composing
-- the input of every component by using previous component outputs. The state
-- of every component is stored at table bptt_data, indexed by time iteration
-- and object name. This states are needed to implement BPTT algorithm.
ann_graph_methods.forward = function(self, input, during_training,
                                     -- this argument is passed by parent
                                     -- ann.graph components for reusability
                                     parent_results)
  self.gradients_computed = false
  forward_asserts(self)
  local bunch_size = get_bunch_size(input)
  ------------------
  -- BPTT section --
  ------------------
  local backstep = self.backstep
  local bptt = self.bptt_data
  if self:get_is_recurrent() then
    -- if it is recurrent, add one to the step counter
    self.bptt_step = self.bptt_step + 1
    if self.bptt_step > self.backstep then self.bptt_step = 1 end
  else
    -- in other case, force a step counter with value 1 and ignore backstep
    self.bptt_step = 1
    backstep = 1
  end
  local input_name  = self.input_name
  local output_name = self.output_name
  local bptt_step   = self.bptt_step
  -- Results table is the place where states will be stored. This method uses a
  -- new table if none is given at the function call.
  local results = parent_results or {}
  -- current time iteration result is initialized with default values for input
  -- object
  results[input_name] = { input = input, output = input }
  -- This auxiliary function receives an object and its delay, and returns the
  -- output activation of the object at the corresponding time iteration. In
  -- case of nil activation, a default matrix with zeroes will be returned.
  local function retrieve_output(obj, delay)
    assert(obj)
    local states = results -- by default, use the results table as output
    if delay ~= 0 then
      -- in case of delayed connection, look for the states at bptt table into
      -- the proper position modulus backstep
      local pos = bptt_step - delay
      if backstep < math.huge then pos = (pos - 1) % backstep + 1 end
      states = bptt[pos] or empty_table
    end
    local value = (states[name_of(obj)] or empty_table).output
    assert(delay > 0 or value) -- sanity check for non-delayed connections
    local sz = (obj == input_name and self:get_output_size()) or obj:get_output_size()
    april_assert(sz > 0,
                 "Unable to initialize or retrieve default activation of component %s (%s)",
                 name_of(obj), type(obj))
    return value or matrix(bunch_size, sz):zeros()
  end
  ------------------
  -- traverse all the components in topological order
  for _,obj in ipairs(self.order) do
    local node   = self.nodes[obj]
    local input  = compose(node_all_in_edges_it(node), retrieve_output)
    local output
    if not class.is_a(obj, ann.graph) then
      output = obj:forward(input, during_training)
    else
      output = obj:forward(input, during_training, results)
    end
    -- copy state of ALL compounds of obj, it can be a complex object
    results = obj:copy_state(results)
  end
  local node   = self.nodes[output_name]
  local output = compose(node_all_in_edges_it(node), retrieve_output)
  forward_finish(self, input, output)
  ------------------
  -- BPTT section --
  ------------------
  results[output_name] = { input=output, output=output }
  -- in case not parent_results given, this component is the parent one, store
  -- the states in the bptt data table
  if not parent_results then bptt[bptt_step] = results end
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
local function ann_graph_backprop(self, time, is_a_child)
  local input_name  = self.input_name
  local output_name = self.output_name
  local bptt        = self.bptt_data
  local bptt_step   = time or self.bptt_step
  local backstep    = self.backstep
  local stop_at     = 1
  -- FIXME: the following assert is only valid for stand-alone graphs
  -- assert(bptt_step == 1 or self:get_is_recurrent(),
  -- "Unable to use BPTT in non recurrent networks")
  backprop_asserts(self)
  -- BPTT SECTION --
  if not self:get_is_recurrent() then
    backstep,stop_at = 1,bptt_step
  elseif is_a_child then
    stop_at=bptt_step
  end
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
  for i=bptt_step,stop_at,-1 do
    local input = bptt[i][output_name].error_output
    -- accumulate error deltas of nodes connected to the output
    accumulate_error_output(self.nodes[output_name], input, i, retrieve_state)
    -- loop over components (loop over space)
    for j=#self.order,1,-1 do
      local obj  = self.order[j]
      local node = self.nodes[obj]
      -- set the state of obj at time iteration i
      obj:set_state(bptt[i])
      local error_input, error_output = bptt[i][name_of(obj)].error_input
      if not class.is_a(obj, ann.graph) then
        if error_input and error_input ~= null_token then
          error_output = obj:backprop(error_input)
        end
      else
        if obj:get_is_recurrent() or (error_input and error_input ~= null_token) then
          error_output = obj:backprop(error_input, i, true)
        end
      end
      obj:copy_state(bptt[i])
      -- accumulate error output into all input connections
      accumulate_error_output(node, error_output, i, retrieve_state)
    end -- for j in reverse topological order
    backprop_finish(self, input, bptt[i][input_name].error_input)
    -- compute and accumulate gradients of current iteration
    ann_graph_compute_gradients(self, bptt, bptt_step)
    -- BPTT SECTION --
    bptt[i][input_name].error_output = bptt[i][input_name].error_input or null_token
    ------------------
  end -- for i in reverse time
  return bptt[stop_at][input_name].error_output
end

-- traverse the graph following inverse topological order (self.order),
-- composing the error input of every component by accumulating previous
-- component error outputs, and storing the error output of every component at
-- the table error_outputs_table
ann_graph_methods.backprop = function(self, input, time, is_a_child)
  local bptt_data   = self.bptt_data
  local bptt_step   = self.bptt_step
  local time        = time or bptt_step
  april_assert(time == bptt_step or self.backstep == math.huge, "%s %s",
               "Backprop for a given time different of last time step only",
               "valid if BPTT truncation = math.huge")
  --
  local bptt_time   = assert(bptt_data[time], "Execute forward before")
  local output_name = self.output_name
  -- keep the backprop input for a future use
  bptt_time[output_name].error_input  = input
  bptt_time[output_name].error_output = input
  if is_a_child then
    return ann_graph_backprop(self, time, true)
  elseif bptt_step == self.backstep or not self:get_is_recurrent() then
    return ann_graph_backprop(self)
  else
    return null_token
  end
end

-- Forces the backpropagation computation, computing the gradients and returning
-- input derivatives for all available time steps (maximum of self.backstep time
-- steps). This method is needed to obtain error deltas at inputs of the graph,
-- and to use this deltas to train other ANNs.
ann_graph_methods.bptt_backprop = function(self, is_a_child)
  local bptt_data  = self.bptt_data
  local bptt_step  = self.bptt_step
  local input_name = self.input_name
  if not rawget(self,"gradients_computed") then
    ann_graph_backprop(self, nil, is_a_child)
    ann_graph_compute_gradients(self)
  end
  if not is_a_child then
    -- return all the error_output at the input of the graph
    return iterator.range(bptt_step):
    map(function(k) return k,assert(bptt_data[k][input_name].error_output) end):table()
  end
end

-- Computes the gradients, forcing to execute backprop in case it is needed.
ann_graph_methods.compute_gradients = function(self, weight_grads, is_a_child)
  if not rawget(self,"gradients_computed") then
    -- ann_graph_backprop implements gradient computation
    ann_graph_backprop(self, nil, is_a_child)
    assert(rawget(self,"gradients_computed"))
  end
  local weight_grads = weight_grads or {}
  for k,v in pairs(self.grads) do weight_grads[k] = v end
  return weight_grads
end

ann_graph_methods.get_bptt_state = function(self, time)
  if not time then return self.bptt_data end
  assert(self.bptt_data[time], "Unable to retrieve the state at the given time")
  return self.bptt_data[time]
end

ann_graph_methods.copy_state = function(self, tbl)
  tbl = tbl or {}
  (ann.components.lua.."copy_state")(self, tbl)
  for _,obj in ipairs(self.order) do obj:copy_state(tbl) end
  tbl[self:get_name()] = { input = self:get_input(),
                           output = self:get_output(),
                           error_input = self:get_error_input(),
                           error_output = self:get_error_output(), }
  return tbl
end

ann_graph_methods.set_state = function(self, tbl)
  (ann.components.lua.."set_state")(self, tbl)
  for _,obj in ipairs(self.order) do obj:set_state(tbl) end
  local data = tbl[self:get_name()]
  if data.input then self:set_input(data.input) end
  if data.output then self:set_output(data.output) end
  if data.error_input then self:set_error_output(data.error_input) end
  if data.error_output then self:set_error_output(data.error_output) end
  return self
end

-- Sets to zero all the intermediate data, even the BPTT stuff.
ann_graph_methods.reset = function(self, n, bptt_data)
  -- change bptt_data by the given one or by a new empty table
  self.bptt_data = bptt_data or {}
  self.bptt_step = 0
  self.grads     = {}
  local bptt_data = self.bptt_data
  for _,obj in ipairs(self.order) do
    if class.is_a(obj, ann.graph) then
      obj:reset(n, bptt_data)
    else
      obj:reset(n)
    end
  end
  (ann.components.lua.."reset")(self, n)
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
  ext_order[#ext_order+1] = self.input_name
  ext_order[#ext_order+1] = self.output_name
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
    (self.backprop==math.huge) and "math.huge" or tostring(self.backstep), ")",
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
  assert(self:get_is_built(), "Execution of build method is needed before")
  assert(backstep > self.max_delay, "Unable to set the given BPTT truncation")
  self.backstep = backstep
  for _,obj in ipairs(self.order) do
    if class.is_a(obj, ann.graph) then
      obj:set_bptt_truncation(backstep)
    end
  end
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
  return iterator(tbl):map(bind(math.div, nil, n)):table()
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
  return iterator(tbl):map(bind(math.div, nil, 2)):table()
end

---------------------------------------------------------------------------

----------------------
-- ANN GRAPH BLOCKS --
----------------------

ann.graph.blocks = {}

-----------------------------------------------------------------------------

ann.graph.blocks.elman = april_doc{
  class = "function",
  summary = "Returns an Elman block implemented using an ann.graph instance",
  params = {
    name = "The name of the component [optional]",
    input = "Size of its input",
    output = "Size of its output",
    actf = "A string with the activation function name [optional], by default it is logistic",
  },
  outputs = { "An instance of ann.graph properly configured to be an Elman layer", }
} ..
  function(tbl)
    local params = get_table_fields({
        name = { type_match="string" },
        input = { type_match="number", mandatory=true },
        output = { type_match="number", mandatory=true },
        actf = { type_match="string", mandatory=true, default="logistic" },
                                    }, tbl)
    local input   = params.input
    local output  = params.output
    local g       = ann.graph(params.name)
    local name    = params.name or g:get_name()
    local layer   = ann.components.hyperplane{
      input=input, output=output,
      bias_weights = "%s::b"%{ name },
      dot_product_weights = "%s::w"%{ name },
      name = name .. "::layer",
      bias_name = "%s::b"%{ name },
      dot_product_name = "%s::w"%{ name },
    }
    local actf    = assert(ann.components.actf[params.actf],
                           "Incorrect actf parameter"){ name = "%s::actf"%{name} }
    local context = ann.components.hyperplane{
      input=output, output=output,
      bias_weights = "%s::context::b"%{ name },
      dot_product_weights = "%s::context::w"%{ name },
      name = name .. "::context::layer",
      bias_name = "%s::context::b"%{ name },
      dot_product_name = "%s::context::w"%{ name },
    }
    local rec_add = ann.graph.add{ input=output*2, output=output,
                                   name = name .. "::memory" }
    g:connect( 'input', layer, rec_add, actf, 'output' )
    g:delayed( actf, context )
    g:connect( context, rec_add )
    return g
  end

-----------------------------------------------------------------------------

ann.graph.blocks.lstm = april_doc{
  class = "function",
  summary = "Returns a LSTM block implemented using an ann.graph instance",
  description = {
    "The returned LSTM has the following characteristics: (1) the gates",
    "receive the input of the component and the peephole; (2) peepholes are",
    "build taking the internal activation state of the memory cell; (3) the",
    "peephole is 1 time instant delayed for input and forget gate, but it",
    "is not delayed for output gate; (4) all gates have logistic activations.",
    "This object declare a bunch of components with the given name as prefix",
    "(or an automatically generated one).",
  },
  params = {
    name = "The name of the component [optional]",
    input = "Size of its input",
    output = "Size of its output",
    actf = "A string with the activation function name [optional], by default it is logistic",
    peepholes = "A boolean [optional], by default it is true",
    input_gate = "A boolean [optional], by default it is true",
    forget_gate = "A boolean [optional], by default it is true",
    output_gate = "A boolean [optional], by default it is true",
  },
  outputs = { "An instance of ann.graph properly configured to be a LSTM", }
} ..
  function(tbl, g, input_component, output_component)
    local params = get_table_fields({
        name = { type_match="string" },
        input = { type_match="number", mandatory=true },
        output = { type_match="number", mandatory=true },
        actf = { type_match="string", default="logistic" },
        peepholes = { default=true },
        input_gate = { default=true },
        output_gate = { default=true },
        forget_gate = { default=true },
                                    }, tbl)
    local input_component  = input_component or 'input'
    local output_component = output_component or 'output'
    local input, output = params.input, params.output
    local g     = g or ann.graph(params.name)
    local name  = params.name or g:get_name()
    local layer = ann.components.hyperplane{
      input=input, output=output,
      bias_weights = "%s::b"%{ name },
      dot_product_weights = "%s::w"%{ name },
      name = name .. "::cell_input",
      bias_name = "%s::b"%{ name },
      dot_product_name = "%s::w"%{ name },
    }
    local actf  = assert(ann.components.actf[params.actf],
                         "Incorrect actf parameter"){ name = name .. "::actf" }
    g:connect( input_component, layer )
    -- recurrent junction component
    local rec_add = ann.graph.add{ input=output*2, output=output,
                                   name = name .. "::memory" }
    --
    local gating_input_size = (params.peepholes and (input+output)) or input
    -- returns an hyperplane for gating
    local function gating(gate_name)
      return ann.components.hyperplane{
        input = gating_input_size, output = output,
        bias_weights = "%s::%s::b"%{ name, gate_name },
        dot_product_weights = "%s::%s::w"%{ name, gate_name },
        name = "%s::%s::layer"%{ name, gate_name },
        bias_name = "%s::%s::b"%{ name, gate_name },
        dot_product_name = "%s::%s::w"%{ name, gate_name },
      }
    end
    -- returns the gate
    local function gate(gate_name)
      return ann.graph.cmul{ input=2*output, output=output,
                             name = "%s::%s::gate" %{ name, gate_name } }
    end
    -- returns the gate activation
    local function gating_actf(gate_name)
      return ann.components.actf.logistic{
        name="%s::%s::actf"%{ name, gate_name }
      }
    end
    -- returns the peephole
    local function peephole(gate_name)
      return ann.graph.bind{ size=gating_input_size,
                             name = "%s::%s::peephole"%{ name, gate_name } }
    end
    -- input gate
    if params.input_gate then
      local input_gating = gating('i')
      local input_gate   = gate('i')
      g:connect( input_gating, gating_actf('i'), input_gate )
      g:connect( layer, input_gate, rec_add )
      if params.peepholes then
        local input_peephole = peephole('i')
        g:connect( input_component, input_peephole, input_gating )
        g:delayed( rec_add, input_peephole )
      else
        g:connect( input_component, input_gating )
      end
    else
      g:connect( layer, rec_add )
    end
    g:connect( rec_add, actf )
    -- output gate
    if params.output_gate then
      local output_gating = gating('o')
      local output_gate   = gate('o')
      g:connect( output_gating, gating_actf('o'), output_gate )
      g:connect( actf, output_gate, output_component )
      if params.peepholes then
        local output_peephole = peephole('o')
        g:connect( input_component, output_peephole, output_gating )
        g:connect( rec_add, output_peephole )
      else
        g:connect( input_component, output_gating )
      end
    else
      g:connect( actf, output_component )
    end
    -- forget gate
    if params.forget_gate then
      local forget_gating = gating('f')
      local forget_gate   = gate('f')
      g:connect( forget_gating, gating_actf('f'), forget_gate, rec_add )
      g:delayed( rec_add, forget_gate )
      if params.peepholes then
        local forget_peephole = peephole('f')
        g:connect( input_component, forget_peephole, forget_gating )
        g:delayed( rec_add, forget_peephole )
      else
        g:connect( input_component, forget_gating )
      end
    else
      g:delayed( rec_add, rec_add )
    end
    --
    return g
  end

---------------------------------------------------------------------------
---------------------------------------------------------------------------
---------------------------------------------------------------------------

ann.graph.test = function()
  local m = matrix(10,20)
  local function check(m, sz)
    utest.check.eq( get_bunch_size(m), sz )
    utest.check.eq( get_bunch_size(tokens.vector.bunch{ m, m }), sz )
    utest.check.eq( get_bunch_size(tokens.vector.bunch{
                                     tokens.vector.bunch{ m },
                                     m }), sz )
  end
  check( matrix(10,20), 10 )
  check( matrix.sparse.diag{1,2,3}, 3)
  --
  utest.check.eq( name_of("one"), "one" )
  utest.check.eq( name_of{ get_name = function() return "one" end }, "one" )
  utest.check.eq( name_of( ann.graph.bind{ name="one" } ), "one" )
  --
  local nodes = { a=node_constructor(), b=node_constructor(),
                  c=node_constructor(), d=node_constructor() }
  node_connect(nodes, "a", "b", 0)
  node_connect(nodes, "a", "c", 0)
  node_connect(nodes, "b", "c", 0)
  node_connect(nodes, "b", "d", 0)
  node_connect(nodes, "c", "d", 0)
  --
  utest.check.eq( nodes.a.out_edges[1], "b" )
  utest.check.eq( nodes.a.out_edges[2], "c" )
  utest.check.eq( nodes.b.out_edges[1], "c" )
  utest.check.eq( nodes.b.out_edges[2], "d" )
  utest.check.eq( nodes.c.out_edges[1], "d" )
  utest.check.eq( #nodes.d.out_edges, 0 )
  --
  utest.check.eq( #nodes.a.in_edges, 0 )
  utest.check.eq( nodes.b.in_edges[1], "a" )
  utest.check.eq( nodes.c.in_edges[1], "a" )
  utest.check.eq( nodes.c.in_edges[2], "b" )
  utest.check.eq( nodes.d.in_edges[1], "b" )
  utest.check.eq( nodes.d.in_edges[2], "c" )
  --
  local nodes = {
    a = { out_edges = { 'b', 'c' }, out_delay_values = { 0, 0 } },
    b = { out_edges = { 'c', 'd' }, out_delay_values = { 0, 0 } },
    c = { out_edges = { 'd' }, out_delay_values = { 0 } },
    d = { out_edges = { }, out_delay_values = { } },
  }
  --
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
