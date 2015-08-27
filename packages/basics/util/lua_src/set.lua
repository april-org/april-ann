local set,set_methods = class("set")
_G.set = set

function set:constructor(tbl)
  self:clear()
  tbl = tbl or {}
  for k,v in ipairs(tbl) do self:add(v) end
end

function set_methods:add(v)
  local tv = type(v)
  assert(tv == "number" or tv == "string", "Only numbers or strings allowed")
  local data = self.data
  local k    = data[v]
  if not k then
    self.n = self.n+1
    k = self.n
    data[v] = k
  end
  return k
end

function set_methods:clear()
  self.n = 0
  self.data = {}
  return self
end

function set_methods:keys()
  return iterator(pairs(self.data)):select(1):table()
end

function set_methods:clone()
  return set(self:ctor_params())
end

function set_methods:ctor_name()
  return "set"
end

function set_methods:ctor_params()
  return self:keys()
end

function set_methods:discard(v)
  local data = self.data
  local k    = data[v]
  if k then
    data[v] = nil
    self.n = self.n - 1
  end
  return self
end

function set_methods:difference(...)
  local others = table.pack(...)
  local result = {}
  for v,k in pairs(self.data) do
    local found = false
    for i=1,#others do
      if others[i].data[v] then found = true break end
    end
    if not found then result[#result+1] = v end
  end
  return set(result)
end

function set_methods:difference_update(...)
  local others = table.pack(...)
  for v,k in pairs(self.data) do
    for i=1,#others do
      if others[i].data[v] then self:remove(v) break end
    end
  end
  return self
end

function set_methods:intersection(...)
  local others = table.pack(...)
  local result = {}
  for v,k in pairs(self.data) do
    local fail = false
    for i=1,#others do
      if not others[i].data[v] then fail = true break end
    end
    if not fail then result[#result+1] = v end
  end
  return set(result)
end

function set_methods:intersection_update(...)
  local others = table.pack(...)
  for v,k in pairs(self.data) do
    for i=1,#others do
      if not others[i].data[v] then self:remove(v) break end
    end
  end
  return self
end

function set_methods:isdisjoint(other)
  local disjoint = true
  for v,k in pairs(self.data) do
    if other.data[v] then disjoint = false break end
  end
  return disjoint
end

function set_methods:issubset(other)
  local subset = true
  for v,k in pairs(self.data) do
    if not other.data[v] then subset = false break end
  end
  return subset
end

function set_methods:issuperset(other)
  return other:issubset(self)
end

function set_methods:pop()
  local v = next(self.data)
  self:remove(v)
  return v
end

function set_methods:remove(v)
  assert(self.data[v], "Element not found")
  self.data[v] = nil
  self.n = self.n - 1
end

-- function set_methods:symmetric_difference(...)
-- end

-- function set_methods:symmetric_difference_update(...)
-- end

function set_methods:union(...)
  local result = self:clone()
  return result:update(...)
end

function set_methods:update(...)
  local others = table.pack(...)
  for i=1,#others do
    for v,k in pairs(others[i].data) do
      self:add(v)
    end
  end
  return self
end

function set_methods:consult(v)
  return self.data[v] or false
end

class.extend_metamethod(set, "__tostring",
                        function(self)
                          local t = iterator(pairs(self.data)):select(1):
                            take(10):map(tostring):table()
                          table.sort(t)
                          if #self > 10 then table.insert(t, "...") end
                          return table.concat{"{ ",table.concat(t, ", ")," } ",
                                              "N=", #self }
end)

class.extend_metamethod(set, "__eq",
                        function(self, other)
                          return self:issubset(other) and self:issuperset(other)
end)

class.extend_metamethod(set, "__lt",
                        function(self, other)
                          return self:issubset(other) and #self < #other
end)

class.extend_metamethod(set, "__len",
                        function(self, other)
                          return self.n
end)

class.extend_metamethod(set, "__sub",
                        function(self, other)
                          return self:difference(other)
end)

class.extend_metamethod(set, "__add",
                        function(self, other)
                          return self:union(other)
end)

-- cross-product
-- class.extend_metamethod(set, "__mul",
--                         function(self, other)
--                           return ...
-- end)

-- set partition
-- class.extend_metamethod(set, "__div",
--                         function(self, other)
-- end)
