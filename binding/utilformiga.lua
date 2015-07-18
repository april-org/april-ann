-- Clases auxiliares para calcular el orden topológico, grafo y pila

------------------
-- Little Stack --
------------------

stack = {}
stack.__index = stack
setmetatable(stack, stack)
stack.__tostring = function() return "stack" end

function stack:push(v)
  self.n = self.n+1
  self.data[self.n] = v
end

function stack:top()
  return self.data[self.n]
end

function stack:pop()
  v = self.data[self.n]
  self.data[self.n] = nil
  self.n = self.n-1
  return v
end

function stack:size()
  return self.n
end

function stack:clear()
  self.n    = 0
  self.data = {}
end

function stack:print()
  print ("Stack: ", table.concat(self.data, ",",1,self.n))
end

function stack:__call()
  local obj = { n=0, data={} }
  setmetatable (obj, self)
  return obj
end

function stack.__newindex(t,k,v)
  error("attempt to modify stack class")
end


------------------
-- Little Graph --
------------------

graph = {}
graph.__index = graph
setmetatable(graph, graph)
graph.__tostring = function() return "graph" end

function graph:add_node(node)
  if not self.nodes[node] then
    self.nodes[node] = { connections={}, next={}, back={} }
  end
end

function graph:connect(node1, dest_list)
  if not self.nodes[node1] then
    self.nodes[node1] = { connections={}, next={}, back={} }
    self.n = self.n + 1
  end
  for i,dest_node in pairs(dest_list) do
    if not self.nodes[dest_node] then
      self.nodes[dest_node] = { connections={}, next={}, back={} }
      self.n = self.n + 1
    end
    if not self.nodes[node1].connections[dest_node] then
      self.nodes[node1].connections[dest_node] = true
      table.insert(self.nodes[node1].next, dest_node)
      table.insert(self.nodes[dest_node].back, node1)
    end
  end
end

function graph:get_node(node)
  return self.nodes[node]
end

function graph:size()
  return self.n
end

function graph:nodes_iterator()
  return pairs(self.nodes)
end

function graph:next_iterator(node)
  return pairs(self.nodes[node].next)
end

function graph:back_iterator(node)
  return pairs(self.nodes[node].back)
end

function graph:__call()
  obj = { n=0, nodes={} }
  setmetatable (obj, self)
  return obj
end

--
-- Strongly Connected Componentes and topologic order
--
-- Algorithm extracted from:
--
-- @article{ nuutila94finding,
--   author = "Esko Nuutila and Eljas Soisalon-Soininen",
--   title = "On Finding the Strongly Connected Components in a Directed Graph",
--   journal = "Information Processing Letters",
--   volume = "49",
--   number = "1",
--   pages = "9-14",
--   year = "1994",
--   url = "citeseer.ist.psu.edu/nuutila94finding.html"
-- }

-- recursive function
function graph:visit(node,useful_node)
  self.visited[node] = true
  self.visit_count   = self.visit_count + 1
  self.root[node]    = self.visit_count
  self.indexes[self.visit_count] = node
  self.in_component[node] = false
  self.the_stack:push(node)
  for _,a_node in self:next_iterator(node) do
    if not self.visited[a_node] and useful_node[a_node] then
      self:visit(a_node,useful_node)
    end
    if not self.in_component[a_node] then
      if self.root[a_node] < self.root[node] then
	self.root[node] = self.root[a_node]
      end
    end
  end
  local w
  if self.indexes[self.root[node]] == node then
    repeat
      w = self.the_stack:pop()
      self.in_component[w] = true
    until node == w
  end
end

function graph:search_strongly_cc(useful_node)
  -- set up attributes
  self.visited     = {} -- visited nodes
  self.root        = {} -- root nodes of connected components
  self.in_component= {} -- indicates wheter a node is in a c.c.
  self.indexes     = {} -- index of node in function 'visit' (visit_count)
  self.visit_count = 0  -- current index for function 'visit' (indexes)
  self.the_stack   = stack()
  
  for a_node,_ in self:nodes_iterator() do
    if not self.visited[a_node] and useful_node[a_node] then
      self:visit(a_node,useful_node)
    end
  end
  self.the_stack:clear()
  -- Creamos el grafo de las componentes conexas
  -- y para cada componente el grafo de sus nodes
  local components = {
    components_graph = graph(), -- El grafo de las componentes
    nodes_graph      = {}       -- Un grafo por cada componente con
                                -- la conexión de sus nodes
  }
  for i,v in pairs(self.root) do
    -- Recorremos todos los nodes
    local component_name = self.indexes[v]
    if not components.nodes_graph[component_name] then
      components.nodes_graph[component_name] = graph()
    end
    -- Anyadimos el node
    components.nodes_graph[component_name]:add_node(component_name)
    components.components_graph:add_node(component_name)
    -- Conexiones
    for _,next in self:next_iterator(i) do
      -- Recorremos las aristas of current node
      if self.root[next] == v then
	-- El node destino está en la misma componente
	-- conexa que el origen
	components.nodes_graph[component_name]:connect (i, {next})
      else
	-- Cada node está en una componente distinta
	-- Creamos la conexión entre las dos componentes conexas
	components.components_graph:connect(component_name, 
					    {self.indexes[self.root[next]]})
      end
    end
  end
  -- unset attributes
  self.visited     = nil
  self.root        = nil
  self.in_component= nil
  self.indexes     = nil
  self.visit_count = nil
  self.the_stack   = nil
  --
  return components
end

function graph:get_reverse_top_order (the_node,useful_node)
  for _,node in self:next_iterator(the_node) do
    if not self.visited[node] and useful_node[node] then
      self.visited[node] = true
      self:get_reverse_top_order(node,useful_node)
    end
  end
  table.insert(self.top_order, the_node)
end

function graph:reverse_top_order(useful_node)
  self.visited    = {}
  self.top_order = {}
  for pkg,_ in self:nodes_iterator() do
    if not self.visited[pkg] and useful_node[pkg] then
      self.visited[pkg] = true
      self:get_reverse_top_order(pkg,useful_node)
    end
  end
  local top_order = self.top_order
  self.visited    = nil
  self.top_order  = nil
  return top_order
end

-- function graph.__newindex(t,k,v)
--   error("attempt to modify graph class")
-- end

