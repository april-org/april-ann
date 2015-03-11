------------------------------------------------------------------------------

local setmetatable = setmetatable
local getmetatable = getmetatable
local string = string
local table = table
local math = math
local pairs = pairs
local ipairs = ipairs
local assert = assert

------------------------------------------------------------------------------

-- DEPRECATED

isa = make_deprecated_function("isa",
                               "class.is_a", class.is_a)
class_extension = make_deprecated_function("class_extension",
                                           "class.extend", class.extend)
class_get = make_deprecated_function("class_get",
                                     "class.consult", class.consult)
class_get_value_of_key = make_deprecated_function("class_get_value_of_key",
                                                  "class.consult",
                                                  class.consult)
is_class = make_deprecated_function("is_class",
                                    "class.is_class", class.is_class)

------------------------------------------------------------------------------

-- Clones the function and its upvalues
local function clone_function(func,lookup_table)
  local lookup_table = lookup_table or {}
  -- clone by using a string dump
  local ok,func_dump = pcall(string.dump, func)
  local func_clone = (ok and loadstring(func_dump)) or func
  if func_clone ~= func then
    -- copy upvalues
    local i = 1
    while true do
      local name,value = debug.getupvalue(func,i)
      if not name then break end
      if name == "_ENV" then
        debug.setupvalue(func_clone, i, value)
      else
        debug.setupvalue(func_clone, i, util.clone(value, lookup_table))
      end
      i = i + 1
    end
  end
  return func_clone
end

------------------------------------------------------------------------------

function iscallable(obj)
  local t = luatype(obj)
  return t == "function" or (t == "table" and (getmetatable(obj) or {}).__call)
end

function april_assert(condition, ...)
   if not condition then
     if next({...}) then
       local s,r = pcall(function (...) return(string.format(...)) end, ...)
         if s then
	   error("assertion failed!: " .. r, 2)
         end
      end
     error("assertion failed!", 2)
   end
   return condition
end

------------------------------------------------------------------------------

local LUA_BIND_CPP_PROPERTIES = {}
setmetatable(LUA_BIND_CPP_PROPERTIES, { mode='k' }) -- table with weak keys

function has_lua_properties_table(obj)
  assert(luatype(obj)=="userdata", "This functions only works with userdata")
  return LUA_BIND_CPP_PROPERTIES[obj]
end

function get_lua_properties_table(obj)
  assert(luatype(obj)=="userdata", "This functions only works with userdata")
  LUA_BIND_CPP_PROPERTIES[obj] = LUA_BIND_CPP_PROPERTIES[obj] or {}
  return LUA_BIND_CPP_PROPERTIES[obj]
end

function set_lua_properties_table(obj, t)
  assert(luatype(obj)=="userdata", "This functions only works with userdata")
  LUA_BIND_CPP_PROPERTIES[obj] = t
  return obj
end

------------------------------------------------------------------------------

function get_table_from_dotted_string(dotted_string, create, basetable)
  local create    = create or false
  local basetable = basetable or _G
  local t = string.tokenize(dotted_string, ".")
  if not create and not basetable[t[1]] then return nil end
  basetable[t[1]] = basetable[t[1]] or {}
  local current = basetable[t[1]]
  for i=2,#t do
    if not create and not current[t[i]] then return nil end
    current[t[i]] = current[t[i]] or {}
    current = current[t[i]]
  end
  return current
end

function check_version(major_num,minor_num,commit_num)
  local major_v,minor_v,commit_v = util.version()
  local major_num  = major_num  or major_v
  local minor_num  = minor_num  or minor_v
  local commit_num = commit_num or commit_v
  if major_num == major_v and minor_num == minor_v and commit_num==commit_v then
    return true
  else
    fprintf(io.stderr,
	    "Incorrect version number, expected %d.%d commit %d, found %d.%d commit %d\n",
	    major_num, minor_num, commit_num, major_v, minor_v, commit_v)
    return false
  end
end

-- makes a FAKE wrapper around a class, so you can re-implement the functions
function class_wrapper(obj,wrapper)
  local wrapper = wrapper or {}
  local current = obj
  while (getmetatable(current) and getmetatable(current).__index and
	 not rawequal(getmetatable(current).__index,current)) do
    current = getmetatable(current).__index
    for i,v in pairs(current) do
      if rawget(wrapper,i) == nil then
	if type(v) == "function" then
	  wrapper[i] =
	    function(first, ...)
	      if rawequal(first,wrapper) then
		return obj[i](obj, ...)
	      else
		return obj[i](...)
	      end
	    end -- function
	else -- if type(v) == "function"
	  wrapper[i] = v
	end -- if type(v) == "function" ... else
      end -- if wrapper[i] == nil
    end -- for
  end -- while
  if getmetatable(wrapper) and getmetatable(wrapper).__index then
    if getmetatable(getmetatable(wrapper).__index) then
      error("class_wrapper not works with derived or nil_safe objects")
    else
      setmetatable(getmetatable(wrapper).__index, getmetatable(obj))
    end
  else wrapper = class_instance(wrapper,
				get_table_from_dotted_string(get_object_id(obj)))
  end
  return wrapper
end

function april_list(t)
  for i,v in pairs(t) do print(i,v) end
end

function get_object_id(obj)
  local id = nil
  if getmetatable(obj) then
    id = getmetatable(obj).id
  end
  return id
end

get_object_cls = class.of

function april_print_script_header(arg,file)
  local file = file or io.stdout
  fprintf(file,"# HOST:     %s\n", (io.popen("hostname", "r"):read("*l")))
  fprintf(file,"# DATE:     %s\n", os.date())
  fprintf(file,"# VERSION:  %d.%d COMMIT %s %s\n", util.version())
  if arg and arg[0] then
    fprintf(file,"# CMD:      %s %s\n", arg[0], table.concat(arg, " "))
  end
end

-- auxiliary function for bind
local function merge_unpack(t1, t2, i, j, n, m)
  i,j = i or 1, j or 1
  n,m = n or t1.n, m or t2.n
  if i <= n then
    if t1[i] ~= nil then
      return t1[i],merge_unpack(t1, t2, i+1, j, n, m)
    else
      return t2[j],merge_unpack(t1, t2, i+1, j+1, n, m)
    end
  elseif j <= m then
    return t2[j],merge_unpack(t1, t2, i, j+1, n, m)
  end
end

-- allow to bind arguments to any Lua function (only variadic arguments)
function bind(func, ...)
  local args = table.pack(...)
  return function(...)
    return func(merge_unpack(args, table.pack(...)))
  end
end

-- unpacks together several tables, by recursion (it is limited to short tables)
local function private_munpack(i, j, m, ...)
  local n = select('#', ...)
  i,j = i or 1, j or 1
  if i <= n then
    local t = select(i, ...)
    local m = m or t.n or #t
    if j <= m then
      return t[j],private_munpack(i, j+1, m, ...)
    else
      return private_munpack(i+1, 1, nil, ...)
    end
  end
end
function multiple_unpack(...)
  if select('#', ...) == 1 then
    return table.unpack(...)
  else
    return private_munpack(1, 1, nil, ...)
  end
end

-- http://lua-users.org/wiki/IteratorsTutorial
function multiple_ipairs(...)
  local t = {...}
  local tmp = {...}
  -- if nothing to iterate over just return a dummy iterator
  if #tmp==0 then
    return function() end, nil, nil
  end
  local function mult_ipairs_it(t, i)
    i = i+1
    local all_nil = true
    for j=1,#t do
      local val = t[j][i]
      if val ~= nil then all_nil = false end
      tmp[j] = val
    end
    if all_nil then return nil end
    return i, unpack(tmp)
  end
  return mult_ipairs_it, t, 0
end

function filter(func, ...)
  assert(func, "filter: needs a function as first argument")
  local t,key,value = {}
  for key,value in ... do
    local v = value or key
    if func(v) then table.insert(t,v) end
  end
  return t
end

function map(func, ...)
  if not func then func = function(v) return v end end
  local t,key,value = {}
  for key,value in ... do
    if not value then key,value = #t+1,key end
    local r = func(value)
    if r then t[key] = r end
  end
  return t
end

function map2(func, ...)
  if not func then func = function(k,v) return v end end
  local t,key,value = {}
  for key,value in ... do
    if not value then key,value = #t+1,key end
    local r = func(key,value)
    if r then t[key] = r end
  end
  return t
end

function mapn(func, f, s, v)
  if not func then func = function(k,...) return ... end end
  local t = {}
  local tmp = table.pack(f(s,v))
  while tmp[1] ~= nil do
    t[v] = func(table.unpack(tmp))
    tmp = table.pack(f(s,tmp[1]))
  end
  return t
end

function glob(...)
  local r = {}
  for i,expr in ipairs(table.pack(...)) do
    local f = io.popen("ls -d "..expr)
    for i in f:lines() do table.insert(r,i) end
    f:close()
  end
  return r
end

-- executes in parallel a function, and returns the concatenation of all results
function parallel_foreach(num_processes, list_number_or_iterator, func)
  local tt = type(list_number_or_iterator)
  assert(tt == "number" or tt == "table" or
           class.is_a(list_number_or_iterator, iterator),
         "Needs an APRIL-ANN iterator,  table or number as 2nd argument")
  local data_it
  -- in any case, convert list_number_or_iterator into an iterator
  if tt == "number" then
    data_it = iterator.range(list_number_or_iterator)
  else
    data_it = iterator(list_number_or_iterator)
  end
  if num_processes == 1 then -- special case for only 1 process
    local out = {}
    while true do
      local arg = table.pack(data_it())
      if arg[1] == nil then break end
      out[#out + 1] = util.pack( func(table.unpack(arg)) )
    end
    return out
  else -- general case for N processes
    local outputs = iterator(range(1,num_processes)):
    map(function(idx) return os.tmpname() end):table()
    local id = util.split_process(num_processes)-1
    local f = io.open(outputs[id+1], "w")
    fprintf(f, "return {\n")
    -- traverse all iterator values
    local index = 0
    while true do
      local arg = table.pack(data_it())
      if arg[1] == nil then break end
      index = index + 1
      if (index%num_processes) == id then -- data correspond to current process
        table.insert(arg, id)
        local ret = util.pack( func(table.unpack(arg)) )
        if ret ~= nil then
          fprintf(f,"[%d] = %s,\n", index, util.to_lua_string(ret, "binary"))
        end
      end
    end
    fprintf(f, "}\n")
    f:close()
    -- waits for all childrens
    if id ~= 0 then util.wait() os.exit(0) end
    util.wait()
    -- maps all the outputs to a table
    return iterator(ipairs(outputs)):
      map(function(index,filename)
          local t = util.deserialize(filename)
          os.remove(filename)
          -- multiple outputs from this filename
          for k,v in pairs(t) do coroutine.yield(k,v) end
      end):table()
  end
end

function clrscr()
  io.write("\027[2J")	-- ANSI clear screen
  io.write("\027[H")	-- ANSI home cursor
end

function printf(...)
  io.write(string.format(...))
end

function fprintf(file,...)
  file:write(string.format(...))
end

function check_mandatory_table_fields(fields, t)
  for _,name in ipairs(fields) do
    table.insert(ret, t[name] or error("The "..name.." field is mandatory"))
  end
end

--
--  local params = get_table_fields{
--    begin_token  = { type_match = "string", mandatory = false, default = "<s>"  },
--    end_token    = { type_match = "string", mandatory = false, default = "</s>" },
--    unknown_word = { type_match = "string", mandatory = false, default = "<unk>" },
--    factors = { type_match = "table", mandatory = true,
--		getter = get_table_fields_ipairs{
--		  vocabulary = { isa_match(lexClass), mandatory = true },
--		  layers = { type_match = "table", mandatory = true,
--			     getter = get_table_fields_ipairs{
--			       actf = { type_match = "string", mandatory = true },
--			       size = { type_match = "number", mandatory = true },
--			     },
--		  },
--		},
--    },
--  }
local valid_get_table_fields_params_attributes = { type_match = true,
						   isa_match  = true,
						   mandatory  = true,
						   getter     = true,
						   default    = true }
function get_table_fields(params, t, ignore_other_fields)
  local is_a = class.is_a
  local type = type
  local pairs = pairs
  local ipairs = ipairs
  local luatype = luatype
  --
  local params = params or {}
  local t      = t or {}
  local ret    = {}
  for key,value in pairs(t) do
    if not params[key] then
      if ignore_other_fields then
	ret[key] = value
      else
	error("Unknown field: " .. key)
      end
    end
  end
  for key,data in pairs(params) do
    if params[key] then
      local data = data or {}
      for k,_ in pairs(data) do
	if not valid_get_table_fields_params_attributes[k] then
	  error("Incorrect parameter to function get_table_fields: " .. k)
	end
      end
      -- each param has type_match, mandatory, default, and getter
      local v = t[key]
      if v == nil then v = data.default end
      if v == nil and data.mandatory then
	error("Mandatory field not found: " .. key)
      end
      if v ~= nil and data.type_match and (luatype(v) ~= data.type_match or type(v) ~= data.type_match) then
	if data.type_match ~= "function" or not iscallable(v) then
	  error("Incorrect type '" .. type(v) .. "' for field '" .. key .. "'")
	end
      end
      if v ~= nil and data.isa_match and not is_a(v, data.isa_match) then
	error("Incorrect field isa_match predicate: " .. key)
      end
      if data.getter then v=(t[key]~=nil and data.getter(t[key])) or nil end
      ret[key] = v
    end  -- if params[key] then ...
  end -- for key,data in pairs(params) ...
  return ret
end

function get_table_fields_ipairs(...)
  local arg = table.pack(...)
  return function(t)
    local table = table
    local t   = t or {}
    local ret = {}
    for i,v in ipairs(t) do
      table.insert(ret, get_table_fields(table.unpack(arg), v))
    end
    return ret
  end
end

function get_table_fields_recursive(...)
  local arg = table.pack(...)
  return function(t)
    local t = t or {}
    return get_table_fields(table.unpack(arg), t)
  end
end

---------------------------------------------------------------
------------------------ MATH UTILS ---------------------------
---------------------------------------------------------------

-- log addition
function math.logadd(a,b)
  if a > b then
    return a + math.log1p(math.exp(b-a))
  else
    return b + math.log1p(math.exp(a-b))
  end
end

-- auxiliary function for fast development of reductions
function math.lnot(a)
  assert(a, "Needs one argument, you can use bind function to freeze any arg")
  return not a
end
-- auxiliary function for fast development of reductions
function math.lor(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a or b
end

-- auxiliary function for fast development of reductions
function math.land(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a and b
end

-- auxiliary function for fast development of reductions
function math.ge(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a>=b
end

-- auxiliary function for fast development of reductions
function math.gt(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a>b
end

-- auxiliary function for fast development of reductions
function math.le(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a<=b
end

-- auxiliary function for fast development of reductions
function math.lt(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a<b
end

-- auxiliary function for fast development of reductions
function math.eq(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a==b
end

-- auxiliary function for fast development of reductions
function math.add(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a+b
end

-- auxiliary function for fast development of reductions
function math.sub(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a-b
end

-- auxiliary function for fast development of reductions
function math.mul(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a*b
end

-- auxiliary function for fast development of reductions
function math.div(a,b)
  assert(a and b,
         "Needs one argument, you can use bind function to freeze any arg")
  return a/b
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

function math.median(t, ini, fin)
  local ini,fin = ini or 1, fin or #t
  local len     = fin-ini+1
  local mpos    = math.floor((ini+fin)/2)
  local median  = t[mpos]
  if len % 2 == 0 then
    median = (median + t[mpos+1])/2
  end
  return median
end

-- calcula la media de una tabla, o subtabla
function math.mean(t, ini, fin)
  local ini,fin = ini or 1, fin or #t
  local total=0
  local suma=0
  total = fin - ini + 1
  for i=ini,fin do
    suma = suma + t[i]
  end
  return suma/total,total
end

-- calcula la desviacion tipica de una tabla o subtabla
function math.std(t, ini, fin)
  local ini,fin = ini or 1, fin or #t
  local mean,total = math.mean(t, ini, fin)
  local suma_sqr=0
  for i=ini,fin do
    local value = mean - t[i]
    suma_sqr = suma_sqr + value*value
  end
  return math.sqrt(suma_sqr/(total-1)),total
end

-- computes the sign of a number
function math.sign(v)
  return (v>0 and 1) or (v<0 and -1) or 0
end


---------------------------------------------------------------
------------------------ STRING UTILS -------------------------
---------------------------------------------------------------

getmetatable("").__mod = function(self,t)
  assert(type(t) == "table", "Needs a table as parameter")
  return self:gsub("$(%a[%a%d]*)",t):format(table.unpack(t))
end

function string.truncate(str, columns, prefix)
  local columns = columns - #prefix - 1
  local lines   = string.tokenize(str, "\n")
  local out     = { { } }
  for _,line in ipairs(lines) do
    local words   = string.tokenize(line, " ")
    local size    = 0
    for i,w in ipairs(words) do
      if #w + size > columns then
	size = 0
	table.insert(out, { prefix })
      end
      table.insert(out[#out], w)
      size = size + #w
    end
  end
  for i=1,#out do out[i] = table.concat(out[i], " ") end
  return table.concat(out, "\n")
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

function string.get_path(path_with_filename, sep)
  local sep=sep or'/'
  return path_with_filename:match("(.*"..sep..")") or "./"
end

string.dirname = string.get_path -- synonim

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
  local sep = sep or ' \t\n\r'
  local list = {}
  for token in string.gmatch(str, '[^'..sep..']+') do
    table.insert(list,token)
  end
  return list
end

function string.tokenize_width(str,width)
  local width = width or 1
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

---------------------------------------------------------------
------------------------ TABLE UTILS --------------------------
---------------------------------------------------------------

function table.clear(t)
  for k,v in pairs(t) do t[k] = nil end
end

function table.unpack_on(t, dest, overwrite)
  for i,j in pairs(t) do
    april_assert(overwrite or not dest[i], "Redefinition of key %s", i)
    dest[i] = j
  end
end

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
  return map(f, ipairs(t))
end

function table.map(t,f)
  return map(f, pairs(t))
end

function table.imap2(t,f)
  return map2(f, ipairs(t))
end

function table.map2(t,f)
  return map2(f, pairs(t))
end

function table.reduce(t,f,initial_value)
  return reduce(f, initial_value, ipairs(t))
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


function table.merge(t1,t2)
  local result = util.clone(t1)
  for k,v in pairs(t2) do result[k] = util.clone(v) end
  return result
end

-- Warning: this function makes a DEEP copy of LUA tables, but userdata objects
-- are copied as references
function table.deep_copy(t, lookup_table)
  local copy = {}
  for i,v in pairs(t) do
    if luatype(v) ~= "table" then
      copy[i] = v
    else
      lookup_table = lookup_table or {}
      lookup_table[t] = copy
      if lookup_table[v] then
	copy[i] = lookup_table[v] -- we already copied this table. reuse the copy.
      else
	copy[i] = table.deep_copy(v,lookup_table) -- not yet copied. copy it.
      end
    end
  end
  return copy
end

-----
-- to string
-----

function table.tostring(t,format)
  if t.to_lua_string then return t:to_lua_string(format) end
  local out  = {}
  -- first serialize the array part of the table
  local j=1
  while t[j] ~= nil do
    local v = t[j]
    local value
    local tt = luatype(v)
    if tt == "table" then value = table.tostring(v,format)
    elseif tt == "string" then value = string.format("%q",v)
    elseif tt == "userdata" then
      assert(v.to_lua_string, "Needs to_lua_string method")
      value = v:to_lua_string(format)
    elseif tt == "function" then
      value = util.function_to_lua_string(v,format)
    else value = tostring(v)
    end
    table.insert(out, value)
    j=j+1
  end
  -- extract all keys removing array part
  local keys = iterator(pairs(t)):select(1):
  filter(function(key) local k = tonumber(key) return not k or k<=0 or k>=j end):table()
  -- sort the keys in order to obtain a deterministic result
  table.sort(keys, function(a,b) return tostring(a) < tostring(b) end)
  -- serialize non array part of the table
  for _,i in ipairs(keys) do
    local v = t[i]
    local key
    local value
    if luatype(i) == "string" then
      key = string.format("[%q]=",i)
    elseif tonumber(i) then key = "["..i.."]".."="
    else key = string.format("[%q]=",tostring(i))
    end
    local tt = luatype(v)
    if tt == "table" then value = table.tostring(v,format)
    elseif tt == "string" then value = string.format("%q",v)
    elseif tt == "userdata" then
      assert(v.to_lua_string, "Needs to_lua_string method")
      value = v:to_lua_string(format)
    elseif tt == "function" then
      value = util.function_to_lua_string(v,format)
    else value = tostring(v)
    end
    table.insert(out, key .. value)
  end
  return "{"..table.concat(out,",").."}"
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

-- devuelve el valor maximo de una tabla
function table.argmax(t)
  local max,index = table.max(t)
  return index
end

-- devuelve el valor minimo de una tabla
function table.argmin(t)
  local max,index = table.min(t)
  return index
end

-- converts an unsorted dictionary in an array, throwing away the keys (the
-- order of the array is not determined)
function table.linearize(t)
  local r = {}
  for k,v in pairs(t) do table.insert(r, v) end
  return r
end

table.luainsert = table.luainsert or table.insert
function table.insert(t,...)
  table.luainsert(t,...)
  return t
end

-- values and keys iterators
function table.values(t)
  local key,value
  return function()
    key,value = next(t,key)
    return value
  end
end

function table.ivalues(t)
  local key=0
  return function()
    key = key+1
    local v = t[key]
    -- TODO: check if table has nil gaps
    return v
  end
end

function table.keys(t)
  local key
  return function()
    key = next(t,key)
    return key
  end
end

function table.ikeys(t)
  local key=0
  return function()
    key = key+1
    -- TODO: check if table has nil gaps
    if t[key] then return key end
  end
end

---------------------------------------------------------------
--------------------------- IO UTILS --------------------------
---------------------------------------------------------------

function io.uncommented_lines(filename)
  local f = io.open(filename, "r")
  if filename~=nil and not f then error("Unable to open " .. filename) end
  f = f or io.stdin
  return function()
    local line = nil
    repeat
      line = f:read("*l")
    until not line or not string.match(line, "^%s*#.*$") 
    return line
  end
end

----------------------------------------------------------------------------

function util.unpack(t)
  if type(t) == "table" then
    return table.unpack(t)
  else
    return t
  end
end

function util.pack(...)
  if select('#',...) == 1 then
    return ...
  else
    return table.pack(...)
  end
end

function util.function_setupvalues(func, upvalues)
  for i,value in ipairs(upvalues) do
    debug.setupvalue(func, i, value)
  end
  return func
end

local function char(c) return ("\\%03d"):format(c:byte()) end
local function szstr(s) return ('"%s"'):format(s:gsub("[^ !#-~]", char)) end

function util.function_to_lua_string(func,format)
  --
  local func_dump = string.format("load(%s)", szstr(string.dump(func)))
  local upvalues = {}
  local i = 1
  while true do
    local name,value = debug.getupvalue(func,i)
    if not name then break end
    -- avoid global environment upvalue
    if name ~= "_ENV" then
      upvalues[i] = value
    end
    i = i + 1
  end
  --
  local t = {
    "util.function_setupvalues(",
    func_dump,
    ",",
    table.tostring(upvalues,format),
    ")"
  }
  return table.concat(t, "")
end

-- It clones a data object. Doesn't work if exists loops in tables.
function util.clone(data, lookup_table)
  if data == nil then return nil end
  local lookup_table = lookup_table or {}
  if lookup_table[data] then
    return lookup_table[data]
  else
    local obj
    local tt = type(data)
    if tt == "number" or tt == "string" or tt == "thread" or tt == "boolean" then obj = data
    elseif tt == "function" then obj = clone_function(data, lookup_table)
      -- FIXME: the following elseif condition can end into a loop if the table
      -- has itself as metatable and clone doesn't exists
    elseif data.clone and type(data.clone) == "function" then obj = data:clone()
    elseif luatype(data) == "userdata" then obj = data
    else
      assert(luatype(data) == "table", "Expected a table")
      obj = {}
      for i,v in pairs(data) do
        local clone_i = util.clone(i, lookup_table)
        local clone_v = util.clone(v, lookup_table)
        obj[clone_i] = clone_v
      end
    end
    lookup_table[data] = obj
    return obj
  end
end

function util.to_lua_string(data,format)
  local tt = luatype(data)
  if tt == "table" then
    if data.to_lua_string then
      return data:to_lua_string(format)
    else
      return table.tostring(data,format)
    end
  elseif tt == "string" then
    return string.format("%q", data)
  elseif tt == "function" then
    return util.function_to_lua_string(data,format)
  elseif tt == "userdata" then
    assert(getmetatable(data) and
	     getmetatable(data).__index and
	     getmetatable(data).__index.to_lua_string,
	   "Userdata needs a to_lua_string(format) method")
    return data:to_lua_string(format)
  else
    return tostring(data)
  end
end

------------------------------------------------------------------------------

util.serialize =
  april_doc{
    class = "function",
    summary = "Serializes an object to a filename or a string",
    params = {
      "Any object, table, string, number, ...",
      "A filename [optional], if not given an string would be returned",
      "Format string: 'ascii' or 'binary' [optional]",
    },
    outputs = {
      "In case of nil second argument, this function",
      "returns a string with the serialized object.",
    },
  } ..
  function (data, where, format)
    local version = { util.version() } table.insert(version, os.date())
    local comment = "-- version info { major, minor, commit number, commit hash, date }"
    local version_info = ",\n%s\n%s\n"%{ comment,
                                         util.to_lua_string(version, format) }
    local tw = type(where)
    if tw == "string" then
      local f = io.open(where, "w")
      f:write("return ")
      f:write(util.to_lua_string(data, format))
      f:write(version_info)
      f:close()
    elseif tw == "nil" then
      return string.format("return %s%s", util.to_lua_string(data, format),
                           version_info)
    elseif iscallable(where) then
      where(string.format("return %s%s", util.to_lua_string(data, format),
                          version_info))
    else
      error("Needs a string, a function, or nil as 2nd argument")
    end
  end

------------------------------------------------------------------------------

util.deserialize = 
  april_doc{
    class = "function",
    summary = "Deserializes an object from a filename or a string",
    params = {
      "A string with a filename or a serialized object",
      { "... a variadic list of arguments to be passed to",
        "the object during deserialization", },
    },
    outputs = {
      "A deserialized object",
    },
  } ..
  function (from, ...)
    assert(from, "A string or function is needed as 1st argument")
    if type(from) == "string" then
      if from:find("^[%s]*return[%s{}()]") then
        return util.deserialize(function() return from end)
      else
        return assert(loadfile(from))(...)
      end
    elseif iscallable(from) then
      local f = load(from())
      return f(...)
    else
      error("Needs a string or a function as 1st argument")
    end
  end

------------------------------------------------------------------------------

-------------------
-- DOCUMENTATION --
-------------------

april_set_doc(april_set_doc, {
		class   = "function",
		summary = "This function adds documentation to april",
		description = {
		  "This function builds documentation data structures.",
		  "The documentation can be retrieved by april_help and",
		  "april_dir functions.",
		},
		params = {
		  "A string with the lua value name",
		  { "A table which contains 'class', 'summary', 'description',",
		    "'params' and 'outputs' fields, described below.", },
		  ["class"] = { "The class of lua value: class, namespace,",
				"function, variable" },
		  ["summary"] = { "A string with a brief description of the",
				  "lua value.",
				  "An array of strings is also valid, which",
				  "will be concatenated with ' '" },
		  ["description"] = { "A string with description of the lua",
				      "value.",
				      "An array of strings is also valid, which",
				      "will be concatenated with ' '" },
		  ["params"] = { "A dictionary string=>string, associates to",
				 "each parameter name a description.",
				 "The description string could be a table,",
				 "which will be contatenated with ' '.", },
		  ["outputs"] = { "A dictionary string=>string, associates to",
				  "each output name a description.",
				  "The description string could be a table,",
				  "which will be contatenated with ' '.", },
		}, })

april_set_doc(map,
	      {
		class = "function",
		summary = "An implementation of python map function",
		description = {
		  "This function returns a table which is the result of",
		  "apply a given function to all the elements of a set.",
		  "The set of elements is traversed using iterator functions.",
		  "The table contains as many elements as the set, and only",
		  "keeps the first returned value of the function (or a table",
		  "if the function returns a table).",
		},
		params = {
		  { "A function to be applied to all elements values [optional].",
		    "If not given, the identity function will be used. " },
		  "An iterator function which returns the whole set of elements",
		  "First argument to the iterator (normally the caller object) [optional]",
		  "Second argument [optional]",
		  "..."
		},
		outputs = {
		  { "A table with the key,value pairs. Keys are",
		    "the same as iterator function returns, but values",
		    "are after application of given function",
		  },
		},
	      })

april_set_doc(map2,
	      {
		class = "function",
		summary = "An implementation of python map function",
		description = {
		  "This function returns a table which is the result of",
		  "apply a given function to all the elements of a set.",
		  "The set of elements is traversed using iterator functions.",
		  "The table contains as many elements as the set, and only",
		  "keeps the first returned value of the function (or a table",
		  "if the function returns a table).",
		},
		params = {
		  { "A function to be applied to all element key,value pairs [optional].",
		    "If not given, the identity function will be used. " },
		  "An iterator function which returns the whole set of elements",
		  "First argument to the iterator (normally the caller object) [optional]",
		  "Second argument [optional]",
		  "..."
		},
		outputs = {
		  { "A table with the key,value pairs. Keys are",
		    "the same as iterator function returns, but values",
		    "are after application of given function",
		  },
		}
	      })

april_set_doc(signal.register,
	      {
		class="function",
		summary="Registers execution of Lua function with a given signal",
		description = {
		  "This function fails if the given signal was registered before",
		  "by other function different than this.",
		  "If a nil value is given for the function, the signal will",
		  "be ignored.",
		},
		params = {
		  "A signal number, use the helpers signal.SIG...",
		  "A lua function",
		},
	      })

april_set_doc(signal.release,
	      {
		class="function",
		summary="Releases the Lua function associated with the given signal",
		description = {
		  "This function fails if the given signal was registered before",
		  "by other function different than signal.register(...).",
		},
		params = {
		  "A signal number, use the helpers signal.SIG_...",
		},
})

--------------------------------------------

april_set_doc(util.stopwatch,{
                class = "class",
                summary = "Stopwatch class for time measurement",
})

april_set_doc(util.stopwatch,{
                class = "function",
                summary = "Constructor of stopwatch",
})

april_set_doc(util.stopwatch.."go", {
                class = "method",
                summary = "Starts timer",
})

april_set_doc(util.stopwatch.."stop", {
                class = "method",
                summary = "Stops timer",
})

april_set_doc(util.stopwatch.."reset",{
                class = "method",
                summary = "Resets timer",
})

april_set_doc(util.stopwatch.."read",{
                class = "method",
                summary = "Resets timer",
                outputs = {
                  "Elapsed CPU time in sec.",
                  "Elapsed wall time in sec."
                },
})

april_set_doc(util.stopwatch.."is_on",{
                class = "method",
                summary = "Is on predicate",
                outputs = {
                  "A boolean indicating object status",
                },
})

april_set_doc(util.stopwatch.."clone",{
                class = "method",
                summary = "Deep copy",
                outputs = {
                  "A deep copy of the caller object",
                },
})
