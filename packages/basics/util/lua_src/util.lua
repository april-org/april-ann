-- Esperamos que en el futuro nos permita ser de GRAN ayuda para
-- conocer entradas/salidas de las funciones y metodos C++
function april_help(t)
  local obj = false
  if type(t) == "function" then
    error("The parameter is a function. Nothing to do :(")
  elseif type(t) ~= "table" then
    if getmetatable(t) and getmetatable(t).__index then
      t = getmetatable(t).__index
      obj = true
    else
      error("The parameter is not a table and has not a "..
	    "metatable or has metatable but not a __index field")
    end
  end
  local print_data = function(d) print("\t * " .. d) end
  local classes = {}
  local funcs   = {}
  local names   = {}
  for i,v in pairs(t) do
    if type(v) == "function" then
      table.insert(funcs, i)
    elseif type(v) == "table" then
      if i ~= "meta_instance" then
	if v.meta_instance then
	  table.insert(classes, i)
	else
	  if getmetatable(v) and not getmetatable(v).__call then
	    table.insert(names, i)
	  else
	    table.insert(funcs, i)
	  end
	end
      end
    end
  end
  if #names > 0 then
    print(" -- Names in the namespace")
    for i,v in pairs(names) do
      print_data(v)
    end
    print("")
  end
  if #classes > 0 then
    print(" -- Classes in the namespace")
    for i,v in pairs(classes) do
      print_data(v)
    end
    print("")
  end
  if #funcs > 0 then
    if not obj then
      print(" -- Static Functions")
    else
      print(" -- Object functions (methods or static functions)")
    end
    for i,v in pairs(funcs) do print_data(v) end
    print("")
  end
  if obj then
    while getmetatable(t) and getmetatable(t).__index do
      local superclass_name = getmetatable(t).id
      t = getmetatable(t).__index
      print(" -- Inherited methods from " .. superclass_name)
      for i,v in pairs(t) do
	if type(v) == "function" then
	  print_data(i)
	end
      end
      print("")
    end
  else
    if t.meta_instance and t.meta_instance.__index then
      print(" -- Methods")
      for i,v in pairs(t.meta_instance.__index) do
	if type(v) == "function" then
	  print_data(i)
	end
      end
      print("")
      t = t.meta_instance.__index
      while getmetatable(t) and getmetatable(t).__index do
	local superclass_name = getmetatable(t).id
	t = getmetatable(t).__index
	print(" -- Inherited methods from " .. superclass_name)
	for i,v in pairs(t) do
	  if type(v) == "function" then
	    print_data(i)
	  end
	end
	print("")
      end
    end
  end
end

function parallel_foreach(num_processes, list, func)
  id = util.split_process(num_processes)-1
  for index, value in ipairs(list) do
    if (index%num_processes) == id then
      func(value)
    end
  end
end

-- Redondea un valor real
function math.round(val)
  if val > 0 then
    return math.floor(val + 0.5)
  end
  return -math.floor(-val + 0.5)
end

function math.clamp(value,lower,upper)
  assert(lower<=upper) -- sanity check
  return math.max(lower,math.min(value,upper))
end

-- calcula la media de una tabla, o subtabla
function math.mean(t, ini, fin)
   local total=0
   local suma=0
   if not ini then ini = 1 end
   if not fin then fin = #t end
   total = fin - ini + 1
   for i=ini,fin do
      suma = suma + t[i]
   end
   return suma/total,total
end

-- calcula la desviacion tipica de una tabla o subtabla
function math.std(t, ini, fin)
   local mean,total = math.mean(t, ini, fin)
   local suma_sqr=0
   if not ini then ini = 1 end
   if not fin then fin = #t end
   for i=ini,fin do
      local value = mean - t[i]
      suma_sqr = suma_sqr + value*value
   end
   return math.sqrt(suma_sqr/(total-1)),total
end

function clrscr()
  io.write("\027[2J")	-- ANSI clear screen
  io.write("\027[H")	-- ANSI home cursor
end

function printf(...)
  io.write(string.format(unpack(arg)))
end

function fprintf(file,...)
  file:write(string.format(unpack(arg)))
end

function range(...)
  local inf,step,sup
  if (arg.n == 1) then
    inf = 1
    step = 1
    sup = arg[1]
  else
    if (arg.n == 2) then
      inf = arg[1]
      step = 1
      sup = arg[2]
    else
      inf = arg[1]
      sup = arg[2]
      step = arg[3]
    end
  end
  local t = {}
  for i = inf,sup,step do
    table.insert(t,i)
  end
  return t
end

function string.lines_of(t)
  return string.gmatch(t,"[^\n]+")
end

function string.chars_of_iterator(s,v)
  if v < string.len(s) then
    v = v+1
    return v,string.sub(s,v,v)
  end
end

function string.chars_of(s)
  return string.chars_of_iterator,s,0
end

function string.tokenize(str,sep)
  sep = sep or ' \t'
  local list = {}
  for token in string.gmatch(str, '[^'..sep..']+') do
    table.insert(list,token)
  end
  return list
end

function string.tokenize_width(str,width)
  width = width or 1
  local list = {}
  for i = 1,string.len(str)-width+1,width do
    table.insert(list, string.sub(str,i,i+width-1))
  end
  return list
end

-- function string.split (text, ...)
--   local delimiter = ((arg.n > 0) and arg[1]) or ' '
--   local list = {}
--   local first,last
--   local pos = 1
--   if string.find("", delimiter, 1, true) then 
--     -- this would result in endless loops
--     error("delimiter matches empty string!")
--   end
--   while 1 do
--     first, last = string.find(text, delimiter, pos,true)
--     if first then -- found?
--       table.insert(list, string.sub(text, pos, first-1))
--       pos = last+1
--     else
--       table.insert(list, string.sub(text, pos))
--       return list
--     end
--   end
-- end

-- function string.join (list,...)
--   local delimiter = ((arg.n > 0) and arg[1]) or ' '
--   return table.concat(list,delimiter)
-- end

string.join = table.concat

function table.invert(t)
  local n = {}
  for i,j in pairs(t) do n[j] = i end
  if n[0] ~= nil then -- FIXME: estoy hay que quitarlo
    error ("ATENCION: table.invert(t) => devuelve un codigo 0")
  end
  return n
end

function table.slice(t, ini, fin)
  local aux = {}
  for i=ini,fin do
    table.insert(aux, t[i])
  end
  return aux
end

function table.search_key_from_value(t,value)
  for i,j in pairs(t) do
    if j == value then return i end
  end
end

function table.imap(t,f)
  local n = {}
  for i,j in ipairs(t) do n[i] = f(j) end
  return n
end

function table.map(t,f)
  local n = {}
  for i,j in pairs(t) do n[i] = f(j) end
  return n
end

function table.ifilter(t,f)
  local n = {}
  for i,j in ipairs(t) do
    if f(i,j) then
      table.insert(n, j)
    end
  end
  return n
end

function table.filter(t,f)
  local n = {}
  for i,j in pairs(t) do
    if f(i,j) then
      n[i]=j
    end
  end
  return n
end
      

function table.join(t1,t2)
   local result={}
   local k=1
   if t1 and #t1 > 0 then
      for _,j in ipairs(t1) do
	 table.insert(result, j)
      end
   end
   if t2 and #t2 > 0 then
      for _,j in ipairs(t2) do
	 table.insert(result, j)
      end
   end
   return result
end

function table.copy(t, lookup_table)
 local copy = {}
 for i,v in pairs(t) do
  if type(v) ~= "table" then
   copy[i] = v
  else
   lookup_table = lookup_table or {}
   lookup_table[t] = copy
   if lookup_table[v] then
    copy[i] = lookup_table[v] -- we already copied this table. reuse the copy.
   else
    copy[i] = tcopy(v,lookup_table) -- not yet copied. copy it.
   end
  end
 end
 return copy
end


function glob(...)
  local r = {}
  for i,expr in ipairs(arg) do
    local f = io.popen("ls -d "..expr)
    for i in f:lines() do table.insert(r,i) end
    f:close()
  end
  return r
end

function table.expand(t)
   local result={}
   for i=1,table.getn(t),2 do
      result = table.join(result, range(t[i], t[i+1]))
   end
   return result
end

function table.compact(t)
   local result={}
   local ini=nil
   local fin=nil
   for i,j in pairs(t) do
      if ini == nil then
	 ini = j
	 fin = j
      elseif j-fin > 1 then
	 table.insert(result, ini)
	 table.insert(result, fin)
	 ini = j
	 fin = j
      else fin = j
      end
   end
   table.insert(result, ini)
   table.insert(result, fin)
   return result
end

-----
-- to string
-----

function table.tostring(t)
  local out = {"{"}
  for i,v in pairs(t) do
    local key
    if type(i)=="number" or tonumber(i) then
      table.insert(out,"["..i.."]".."=")
    else
      table.insert(out,string.format("[%q]=",i))
    end
    if type(v) == "table" then
      table.insert(out,"\n"..table.tostring(v))
    elseif type(v) == "string" then
      table.insert(out,string.format("%q",v))
    else
      table.insert(out,tostring(v))
    end
  end
  table.insert(out,"}\n")
  return table.concat(out,",")
end

-- te da una nueva tabla reversa de la anterior
function table.reverse(t)
  local n = {}
  local lenp1 = #t+1
  for i = 1,#t do
    n[lenp1-i] = t[i]
  end
  return n
end

-- devuelve el valor maximo de una tabla
function table.max(t)
  local max,index
  for i,j in pairs(t) do
    if max==nil or j>max then
      index = i
      max   = j
    end
  end
  return max,index
end

-- devuelve el valor minimo de una tabla
function table.min(t)
  local min,index
  for i,j in pairs(t) do
    if min==nil or j<min then
      index = i
      min   = j
    end
  end
  return min,index
end

function string.basename(path)
        local name = string.match(path, "([^/]+)$")
        return name
end

function string.remove_extension(path)
  local name,ext = string.match(path, "(.*)[.]([^.]*)$")
  return name,ext
end

function string.get_extension(path)
        local ext = string.match(path, ".*[.]([^.]*)$")
        return ext
end
