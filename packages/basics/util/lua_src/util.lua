local COLWIDTH=70

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

-- Convert a table in a class, and it receives an optional parent class to
-- implement simple heritance
function class(classname, parentclass)
  local current = get_table_from_dotted_string(classname, true)
  local t = string.tokenize(classname,".")
  if not parentclass then
    current.__index = current
  else
    current.__index = parentclass
  end
  -- To be coherent with C++ binded classes
  current.meta_instance = { __index = current }
  current.id = classname
  setmetatable(current, current)
end

-- Converts a Lua table in an instance of the given class. An optional
-- nil-safe boolean with true indicates if the resulting table field names are
-- nil safe (is not possible to get a field which doesn't exists)
function class_instance(obj, class, nil_safe)
  setmetatable(obj, class)
  if nil_safe then obj.__index = class end
  return obj
end

-- Predicate which returns true if a given object instance is a subclass of a
-- given Lua table (it works for Lua class(...) and C++ binding)
function isa( object_instance, base_class_table )
  local base_class_meta = (base_class_table.meta_instance or {}).__index
  local object_table    = object_instance
  local _isa            = false
  while (not _isa and object_table and getmetatable(object_table) and
	 getmetatable(object_table).__index) do
    local t = getmetatable(object_table).__index
    _isa = (t == base_class_meta)
    object_table = t
  end
  return _isa
end

-- help documentation
local allowed_classes = { ["class"]=true,
			  ["namespace"]=true,
			  ["function"]=true,
			  ["method"]=true,
			  ["var"]=true }
function april_set_doc(table_name, docblock)
  local docblock = get_table_fields(
    {
      class       = { mandatory=true,  type_match="string", default=nil },
      summary     = { mandatory=true },
      description = { mandatory=false, default=docblock.summary },
      params      = { mandatory=false, type_match="table", default=nil },
      outputs     = { mandatory=false, type_match="table", default=nil },
    }, docblock)
  assert(allowed_classes[docblock.class], "Incorrect class: " .. docblock.class)
  _APRIL_DOC_TABLE_ = _APRIL_DOC_TABLE_ or {}
  if type(docblock.summary) == "table" then
    docblock.summary = table.concat(docblock.summary, " ")
  end
  assert(type(docblock.summary) == "string", "Incorrect summary type")
  if type(docblock.description) == "table" then
    docblock.description = table.concat(docblock.description, " ")
  end
  assert(type(docblock.description) == "string", "Incorrect description type")
  if docblock.params then
    for i,v in pairs(docblock.params) do
      if type(v) == "table" then
	docblock.params[i] = table.concat(v, " ")
      end
    end
  end
  if docblock.outputs then
    for i,v in pairs(docblock.outputs) do
      if type(v) == "table" then
	docblock.outputs[i] = table.concat(v, " ")
      end
    end
  end
  local current = get_table_from_dotted_string(table_name, true,
					       _APRIL_DOC_TABLE_)
  table.insert(current, docblock)
end

function april_list(t)
  if type(t) ~= "table" then error("Needs a table") end
  for i,v in pairs(t) do print(i,v) end
end

function april_print_doc(table_name, verbosity, prefix)
  assert(type(table_name)=="string", "Needs a string as first argument")
  assert(type(verbosity)=="number",  "Needs a number as first argument")
  local prefix = prefix or ""
  local current_table
  if #table_name==0 then current_table=_APRIL_DOC_TABLE_
  else
    current_table = get_table_from_dotted_string(table_name, true,
						 _APRIL_DOC_TABLE_)
  end
  if #current_table == 0 then
    if verbosity > 1 then
      print("No documentation found. Check that you are asking for a BASE "..
	      "class method, not a child class inherited method.")
    end
    table.insert(current_table, {})
  end
  local t = string.tokenize(table_name, ".")
  if #t == 0 then table.insert(t, "") end
  for idx,current in ipairs(current_table) do
    if idx > 1 and verbosity > 1 then
      print("\t--------------------------------------------------------------\n")
    end
    local name = table_name
    local out = { }
    if verbosity > 1 then
      table.insert(out,{prefix,
			ansi.fg["bright_red"]..
			  (string.format("%9s",current.class or "")),
			ansi.fg["green"]..table_name..ansi.fg["default"]})
    else
      name = t[#t]
      table.insert(out,
		   {prefix,
		    ansi.fg["bright_red"]..
		      (string.format("%9s",current.class or "")),
		    ansi.fg["green"]..t[#t]..ansi.fg["default"]})
    end
    if verbosity > 0 then
      if current.summary then
	if #name<24 then
	  table.insert(out[1], ansi.fg["cyan"].."=>"..ansi.fg["default"])
	  local aux = "          "
	  local str = string.truncate(current.summary, COLWIDTH,
				      aux..aux..aux)
	  str = string.gsub(str, "%[(.*)%]",
			    "["..ansi.fg["bright_yellow"].."%1"..
			      ansi.fg["default"].."]")
	  table.insert(out[1], str)
	else
	  local aux = "                              "
	  local str = string.truncate(current.summary, COLWIDTH, aux)
	  str = string.gsub(str, "%[(.*)%]",
			    "["..ansi.fg["bright_yellow"].."%1"..
			      ansi.fg["default"].."]")
	  table.insert(out, { aux, str })
	end
      end
    end
    if verbosity > 1 then
      if current.description then
	local str = string.truncate(current.description, COLWIDTH,
				    "            ")
	str = string.gsub(str, "%[(.*)%]",
			  "["..ansi.fg["bright_yellow"].."%1"..
			    ansi.fg["default"].."]")
	table.insert(out,
		     { "\n"..ansi.fg["cyan"].."description:"..ansi.fg["default"],
		       str })
      end
      if current.params then
	table.insert(out,
		     { "\n"..ansi.fg["cyan"].."parameters:"..ansi.fg["default"] })
	local names_table = {}
	for name,_ in pairs(current.params) do table.insert(names_table,name) end
	table.sort(names_table, function(a,b) return tostring(a)<tostring(b) end)
	for k,name in ipairs(names_table) do
	  local description = current.params[name]
	  local str = string.truncate(description, COLWIDTH,
				      "                         ")
	  str = string.gsub(str, "%[(.*)%]",
			    "["..ansi.fg["bright_yellow"].."%1"..
			      ansi.fg["default"].."]")
	  table.insert(out,
		       { "\t",
			 ansi.fg["green"]..string.format("%16s",name)..ansi.fg["default"],
			 str } )
	end
      end
      if current.outputs then
	table.insert(out,
		     { "\n"..ansi.fg["cyan"].."outputs:"..ansi.fg["default"] })
	local names_table = {}
	for name,_ in pairs(current.outputs) do table.insert(names_table,name) end
	table.sort(names_table, function(a,b) return tostring(a)<tostring(b) end)
	for k,name in ipairs(names_table) do
	  local description = current.outputs[name]
	  local str = string.truncate(description, COLWIDTH,
				      "                         ")
	  str = string.gsub(str, "%[(.*)%]",
			    "["..ansi.fg["bright_yellow"].."%1"..
			      ansi.fg["default"].."]")
	  table.insert(out,
		       { "\t",
			 ansi.fg["green"]..string.format("%16s",name)..ansi.fg["default"],
			 str } )
	end
      end
    end
    for i=1,#out do out[i] = table.concat(out[i], " ") end
    print(table.concat(out, "\n"))
    if verbosity > 1 then print("") end
  end
end

function get_object_id(obj)
  local id = nil
  if getmetatable(obj) then
    id = getmetatable(obj).id
  end
  return id
end

-- verbosity => 0 only names, 1 only summary, 2 all
function april_help(table_name, verbosity)
  if not table_name then table_name="" end
  if (type(table_name) ~= "string" and
      getmetatable(table_name) and
      getmetatable(table_name).id) then
    table_name = getmetatable(table_name).id
  end
  assert(type(table_name) == "string", "Expected string as first argument")
  local t
  if #table_name == 0 then t = _G
  else
    t = get_table_from_dotted_string(table_name)
    if not t then
      local aux  = string.tokenize(table_name, ".")
      local last = aux[#aux]
      t = get_table_from_dotted_string(table.concat(aux, ".", 1, #aux-1))
      if not t or not getmetatable(t) then
	error(table_name .. " not found")
      end
      local auxt = getmetatable(t)[last]
      if not auxt then
	if not t.meta_instance then
	  error(table_name .. " not found")
	end
	t = t.meta_instance.__index[last] or error(table_name .. " not found")
      else t = auxt
      end
    end
  end
  local verbosity = verbosity or 2
  local obj = false
  if type(t) == "function" then
    april_print_doc(table_name, verbosity)
    -- printf("No more recursive help for %s\n", table_name)
    return
  elseif type(t) ~= "table" then
    if getmetatable(t) and getmetatable(t).__index then
      local id = getmetatable(t).id
      t = get_table_from_dotted_string(id)
      april_print_doc(id, verbosity)
    else
      april_print_doc(table_name, verbosity)
    end
  else
    april_print_doc(table_name, verbosity)
  end
  -- local print_data = function(d) print("\t * " .. d) end
  local classes    = {}
  local funcs      = {}
  local names      = {}
  local vars       = {}
  for i,v in pairs(t) do
    if type(v) == "function" then
      table.insert(funcs, {i, string.format("%8s",type(v))})
    elseif type(v) == "table" then
      if i ~= "meta_instance" then
	if getmetatable(v) and getmetatable(v).id then
	  if i~="__index" then table.insert(classes, i) end
	else
	  if not getmetatable(v) or not getmetatable(v).__call then
	    table.insert(names, i)
	  else
	    table.insert(funcs, {i, string.format("%8s",type(v))})
	  end
	end
      end
    else
      table.insert(vars, {i, string.format("%8s",type(v))})
    end
  end
  if #vars > 0 then
    print(ansi.fg["cyan"].." -- basic variables (string, number)"..
	ansi.fg["default"])
    table.sort(vars, function(a,b) return a[1] < b[1] end)
    for i,v in pairs(vars) do
      april_print_doc(table_name .. "." .. v[1], math.min(1, verbosity),
		      ansi.fg["cyan"].."   * "..
			v[2]..ansi.fg["default"])
      -- print_data(v)
    end
    print("")
  end
  if #names > 0 then
    print(ansi.fg["cyan"].." -- names in the namespace"..ansi.fg["default"])
    table.sort(names)
    for i,v in pairs(names) do
      april_print_doc(table_name .. "." .. v, math.min(1, verbosity),
		      ansi.fg["cyan"].."   *"..ansi.fg["default"])
      -- print_data(v)
    end
    print("")
  end
  if #classes > 0 then
    print(ansi.fg["cyan"].." -- classes in the namespace"..ansi.fg["default"])
    table.sort(classes)
    for i,v in pairs(classes) do
      april_print_doc(table_name .. "." .. v,
		      math.min(1, verbosity),
		      ansi.fg["cyan"].."   *"..ansi.fg["default"])
      -- print_data(v)
    end
    print("")
  end
  if getmetatable(t) ~= t and #funcs > 0 then
    print(ansi.fg["cyan"].." -- static functions or tables"..ansi.fg["default"])
    table.sort(funcs, function(a,b) return a[1] < b[1] end)
    for i,v in ipairs(funcs) do
      april_print_doc(table_name .. "." .. v[1],
		      math.min(1, verbosity),
		      ansi.fg["cyan"].."   * "..
			v[2]..ansi.fg["default"])
      -- print_data(v)
    end
    print("")
  end
  if t.meta_instance and t.meta_instance.__index then
    print(ansi.fg["cyan"].." -- methods"..ansi.fg["default"])
    local aux = {}
    for i,v in pairs(t.meta_instance.__index) do
      if type(v) == "function" then
	table.insert(aux, i)
      end
    end
    if getmetatable(t) then
      for i,v in pairs(getmetatable(t)) do
	if type(v) == "function" then
	  table.insert(aux, i)
	end
      end
    end
    local prev = nil
    table.sort(aux)
    for i,v in ipairs(aux) do
      if v ~= prev then
	april_print_doc(table_name .. "." .. v,
			math.min(1, verbosity),
			ansi.fg["cyan"].."   *"..ansi.fg["default"])
      end
      prev = v
      -- print_data(v)
    end
    print("")
    t = t.meta_instance.__index
    while (getmetatable(t) and getmetatable(t).__index and
	   getmetatable(t).__index ~= t) do
      local superclass_name = getmetatable(t).id
      t = getmetatable(t).__index
      print(ansi.fg["cyan"]..
	      " -- inherited methods from " ..
	      superclass_name..ansi.fg["default"])
      local aux = {}
      for i,v in pairs(t) do
	if type(v) == "function" then
	  table.insert(aux, i)
	end
      end
      table.sort(aux)
      for i,v in ipairs(aux) do
	april_print_doc(superclass_name .. "." .. v,
			math.min(1, verbosity),
			ansi.fg["cyan"].."   *"..ansi.fg["default"])
	-- print_data(v)
      end
      print("")
    end
  end
  print()
end

function april_dir(t, verbosity)
  april_help(t, 0)
end

function april_print_script_header(arg,file)
  local file = file or io.stdout
  fprintf(file,"# HOST:\t %s\n", (io.popen("hostname", "r"):read("*l")))
  fprintf(file,"# DATE:\t %s\n", (io.popen("date", "r"):read("*l")))
  fprintf(file,"# CMD: \t %s %s\n", arg[0], table.concat(arg, " "))
end

function map(func, ...)
  if not func then func = function(v) return v end end
  local t,key,value = {}
  for key,value in unpack(arg) do
    if not value then key,value = #t+1,key end
    local r = func(value)
    if r then t[key] = r end
  end
  return t
end

function map2(func, ...)
  if not func then func = function(k,v) return v end end
  local t,key,value = {}
  for key,value in unpack(arg) do
    if not value then key,value = #t+1,key end
    local r = func(key,value)
    if r then t[key] = r end
  end
  return t
end

function reduce(func, initial_value, ...)
  assert(type(func) == "function", "Needs a function as first argument")
  assert(initial_value ~= nil,
	 "Needs an initial_value as second argument")
  local accum,key,value = initial_value
  for key,value in unpack(arg) do
    accum = func(accum, value or key)
  end
  return accum
end

-- This function prepares a safe environment for call user functions
function safe_call(f, env, ...)
  env = env or {}
  env.os         = nil    env.io        = nil     env.file     = nil
  env.debug      = nil    env.load      = nil     env.loadfile = nil
  env.load       = nil    env.dofile    = nil     env.math     = math
  env.table      = table  env.string    = string  env.tonumber = tonumber
  env.loadstring = nil    env.courutine = nil     env.print    = print
  env.pairs      = pairs  env.ipairs    = ipairs  env.tostring = tostring
  env.printf     = printf
  env.io = { stderr = io.stderr,
	     stdout = io.stdout }
  setfenv(f, env)
  local status,result_or_error = pcall(f, unpack(arg))
  if not status then
    print(result_or_error)
    error("Incorrect function call")
  end
  return result_or_error
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

function parallel_foreach(num_processes, list, func)
  id = util.split_process(num_processes)-1
  for index, value in ipairs(list) do
    if (index%num_processes) == id then
      func(value)
    end
  end
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
  local inf,sup,step = arg[1],arg[2],arg[3] or 1
  local i = inf - step
  return function()
    i = i + step
    if i <= sup then return i end
  end
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
function get_table_fields(params, t)
  local ret = {}
  for key,value in pairs(t) do
    if not params[key] then error("Unknown field: " .. key) end
  end
  for key,data in pairs(params) do
    local data = data or {}
    for k,_ in pairs(data) do
      if not valid_get_table_fields_params_attributes[k] then
	error("Incorrect parameter to function get_table_fields: " .. k)
      end
    end
    -- each param has type_match, mandatory, default, and getter
    local v = t[key] or data.default
    if v == nil and data.mandatory then
      error("Mandatory field not found: " .. key)
    end
    if v ~= nil and data.type_match and type(v) ~= data.type_match then
      error("Incorrect field type: " .. key)
    end
    if v ~= nil and data.isa_match and not isa(v, data.isa_match) then
      error("Incorrect field isa_match predicate: " .. key)
    end
    if data.getter then v=(t[key]~=nil and data.getter(t[key])) or nil end
    ret[key] = v
  end
  return ret
end

function get_table_fields_ipairs(...)
  return function(t)
    local ret = {}
    for i,v in ipairs(t) do
      table.insert(ret, get_table_fields(unpack(arg), v))
    end
    return ret
	 end
end

function get_table_fields_recursive(...)
  return function(t)
    return get_table_fields(unpack(arg), t)
  end
end

---------------------------------------------------------------
------------------------ MATH UTILS ---------------------------
---------------------------------------------------------------

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

---------------------------------------------------------------
------------------------ STRING UTILS -------------------------
---------------------------------------------------------------

function string.truncate(str, columns, prefix)
  local columns = columns - #prefix - 1
  local words   = string.tokenize(str, " ")
  local out     = { { } }
  local size    = 0
  for i,w in ipairs(words) do
    if #w + size > columns then
      size = 0
      table.insert(out, { prefix })
    end
    table.insert(out[#out], w)
    size = size + #w
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
  sep=sep or'/'
  return path_with_filename:match("(.*"..sep..")") or ""
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

---------------------------------------------------------------
------------------------ TABLE UTILS --------------------------
---------------------------------------------------------------

function table.clear(t)
  for k,v in pairs(t) do t[k] = nil end
end

function table.unpack_on(t, dest)
  for i,j in pairs(t) do
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
  return map(f, ipairs, t)
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

-- Warning: this function makes a DEEP copy of LUA tables, but userdata objects
-- are copied as references
function table.deep_copy(t, lookup_table)
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
    copy[i] = table.deep_copy(v,lookup_table) -- not yet copied. copy it.
   end
  end
 end
 return copy
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

---------------------------------------------------------------
--------------------------- IO UTILS --------------------------
---------------------------------------------------------------

function io.uncommented_lines(filename)
  local f = (filename and io.open(filename, "r")) or io.stdin
  return function()
    local line = nil
    repeat
      line = f:read("*l")
    until not line or not string.match(line, "^%s*#.*$") 
    return line
	 end
end

-------------------
-- DOCUMENTATION --
-------------------
april_set_doc("class", {
		class = "function",
		summary = "This function creates a lua class table",
		description = {
		  "Creates a lua class table for the given",
		  "dotted table name string. Also it is possible to",
		  "especify a parentclass for simple hieritance.", },
		params = {
		  "The table name string",
		  "The parent class table [optional]",
		}, })

april_set_doc("class_instance", {
		class = "function",
		summary = "This function makes a table the instance of a class",
		description = {
		  "Transforms a table to be the instance of a given class.",
		  "It supports an optional argument to indicate if the instance",
		  "is nonmutable, so the user can't create new indexes.", },
		params = {
		  "The table object",
		  "The class table",
		  { "A boolean indicating if it is nil-safe [optional], by",
		    "default it is false. If true, nil fields will throw",
		    "error" },
		},
		outputs = {
		  "The table instanced as object of the given class",
		}, })

april_set_doc("isa", {
		class = "function",
		summary = "A predicate to check if a table is instance of a class",
		params = {
		  "The table object",
		  "The class table",
		},
		outputs = {
		  "A boolean",
		}, })

april_set_doc("april_set_doc", {
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

april_set_doc("map",
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

april_set_doc("map2",
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
