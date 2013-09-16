local COLWIDTH=70

function class_get_value_of_key(class_table, key)
  if class_table.meta_instance and
  luatype(class_table.meta_instance.__index) == "table" then
    return class_table.meta_instance.__index[key]
  else
    error("The table is not a class")
  end
end

function class_extension(class_table, key, value)
  if class_table.meta_instance and
  luatype(class_table.meta_instance.__index) == "table" then
    class_table.meta_instance.__index[key] = value
  else
    error("The table is not a class")
  end
end

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

function check_version(major_num,minor_num)
  local major_v,minor_v = util.version()
  if major_num == major_v and minor_num == minor_v then return true
  else
    fprintf(io.stderr,
	    "Incorrect version number, expected %d.%d, found %d.%d",
	    major_num, minor_num, major_v, minor_v)
  end
end

-- makes a FAKE wrapper around a class, so you can re-implement the functions
function class_wrapper(obj,wrapper)
  local wrapper = wrapper or {}
  local current = obj
  while (getmetatable(current) and getmetatable(current).__index and
	 getmetatable(current).__index ~= current) do
    current = getmetatable(current).__index
    for i,v in pairs(current) do
      if wrapper[i] == nil then
	if type(v) == "function" then
	  wrapper[i] =
	    function(first, ...)
	      if first == wrapper then
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
  else wrapper = class_instance(wrapper, getmetatable(obj))
  end
  return wrapper
end

-- Convert a table in a class, and it receives an optional parent class to
-- implement simple heritance. It returns
-- a table which will contain the methods of the object, and the metatable
-- of the class, so in the metatable could be defined __call constructor.
function class(classname, parentclass)
  local current = get_table_from_dotted_string(classname, true)
  if type(parentclass) == "string" then
    parentclass = get_table_from_dotted_string(parentclass)
  end
  assert(parentclass==nil or is_class(parentclass),
	 "The parentclass must be defined by 'class' function")
  local t = string.tokenize(classname,".")
  --
  local meta_instance = {
    id = classname,
    __tostring = function(self) return "instance of " .. classname end,
    __index = { }
  }
  local class_metatable = {
    __tostring = function() return "class ".. classname .. " class" end,
    id         = classname .. " class",
  }
  if parentclass then
    setmetatable(meta_instance.__index, parentclass.meta_instance)
  end
  -- 
  current.meta_instance = meta_instance
  setmetatable(current, class_metatable)
  return current.meta_instance.__index,class_metatable
end

-- Converts a Lua table in an instance of the given class. An optional
-- nil-safe boolean with true indicates if the resulting table field names are
-- nil safe (is not possible to get a field which doesn't exists)
function class_instance(obj, class, nil_safe)
  setmetatable(obj, class.meta_instance)
  if nil_safe and not getmetatable(class.meta_instance.__index) then
    setmetatable(class.meta_instance, class.meta_instance)
  end
  return obj
end

-- returns true if the given table is a class table (not instance)
function is_class(t)
  return luatype(t) == "table" and t.meta_instance and t.meta_instance.id ~= nil
end

-- Predicate which returns true if a given object instance is a subclass of a
-- given Lua table (it works for Lua class(...) and C++ binding)
function isa( object_instance, base_class_table )
  assert(luatype(base_class_table) == "table",
	 "The second argument must be a class table")
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

-- calls the method of the directly super class
function super(self, methodname, ...)
  local super_instance = getmetatable(self).__index
  assert(super_instance, "The given object hasn't a super-class")
  local aux = super_instance[methodname]
  assert(aux~=nil, "Method " .. methodname .. " not found")
  return super_instance[methodname](...)
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
  if luatype(docblock.summary) == "table" then
    docblock.summary = table.concat(docblock.summary, " ")
  end
  assert(luatype(docblock.summary) == "string", "Incorrect summary type")
  if luatype(docblock.description) == "table" then
    docblock.description = table.concat(docblock.description, " ")
  end
  assert(luatype(docblock.description) == "string", "Incorrect description type")
  if docblock.params then
    for i,v in pairs(docblock.params) do
      if luatype(v) == "table" then
	docblock.params[i] = table.concat(v, " ")
      end
    end
  end
  if docblock.outputs then
    for i,v in pairs(docblock.outputs) do
      if luatype(v) == "table" then
	docblock.outputs[i] = table.concat(v, " ")
      end
    end
  end
  local current = get_table_from_dotted_string(table_name, true,
					       _APRIL_DOC_TABLE_)
  table.insert(current, docblock)
end

function april_list(t)
  if luatype(t) ~= "table" then error("Needs a table") end
  for i,v in pairs(t) do print(i,v) end
end

function april_print_doc(table_name, verbosity, prefix)
  assert(type(table_name)=="string", "Needs a string as first argument")
  assert(type(verbosity)=="number",  "Needs a number as first argument")
  local prefix = prefix or ""
  local current_table
  if #table_name==0 then current_table=_APRIL_DOC_TABLE_
  else
    table_name=table_name:gsub(" class","")
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
  if (luatype(table_name) ~= "string" and
      get_object_id(table_name)) then
    table_name = get_object_id(table_name):gsub(" class","")
  end
  assert(luatype(table_name) == "string",
	 "Expected string as first argument")
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
  if luatype(t) == "function" then
    april_print_doc(table_name, verbosity)
    -- printf("No more recursive help for %s\n", table_name)
    return
  elseif is_class(t) then
    local id = get_object_id(t):gsub(" class","")
    t = get_table_from_dotted_string(id)
    april_print_doc(id, verbosity)
  else
    april_print_doc(table_name, verbosity)
  end
  -- local print_data = function(d) print("\t * " .. d) end
  local classes    = {}
  local funcs      = {}
  local names      = {}
  local vars       = {}
  for i,v in pairs(t) do
    if is_class(v) then
      local id = get_object_id(v):gsub(" class","")
      table.insert(classes, i)
    elseif luatype(v) == "function" or (luatype(v) == "table" and v.__call) then
      table.insert(funcs, {i, string.format("%8s",luatype(v))})
    elseif luatype(v) == "table" then
      table.insert(names, i)
    else
      table.insert(vars, {i, string.format("%8s",luatype(v))})
    end
  end
  if #vars > 0 then
    print(ansi.fg["cyan"].." -- basic variables (string, number)"..
	ansi.fg["default"])
    table.sort(vars, function(a,b) return tostring(a[1]) < tostring(b[1]) end)
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
  if #funcs > 0 then
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
      if luatype(v) == "function" then
	table.insert(aux, i)
      end
    end
    if getmetatable(t) then
      for i,v in pairs(getmetatable(t)) do
	if luatype(v) == "function" then
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
      local superclass_name = getmetatable(t).id:gsub(" class","")
      t = getmetatable(t).__index
      print(ansi.fg["cyan"]..
	      " -- inherited methods from " ..
	      superclass_name..ansi.fg["default"])
      local aux = {}
      for i,v in pairs(t) do
	if luatype(v) == "function" then
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
  fprintf(file,"# DATE:\t %s\n", os.date())
  fprintf(file,"# CMD: \t %s %s\n", arg[0], table.concat(arg, " "))
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

function iterable_filter(func, f, s, v)
  return function(s,v)
    local tmp = table.pack(f(s,v))
    while tmp[1] ~= nil and not func(table.unpack(tmp)) do
      v = tmp[1]
      tmp = table.pack(f(s,v))
    end
    return table.unpack(tmp)
  end, s, v
end

-- FROM: http://www.corsix.org/content/mapping-and-lua-iterators
function iterable_map(func, f, s, v)
  local done
  local function maybeyield(...)
    if ... ~= nil then
      coroutine.yield(...)
    end
  end
  local function domap(...)
    v = ...
    if v ~= nil then
      return maybeyield(func(...))
    else
      done = true
    end
  end
  return coroutine.wrap(function()
			  repeat
			    domap(f(s,v))
			  until done
			end)
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

function reduce(func, initial_value, ...)
  assert(initial_value ~= nil,
	 "reduce: needs an initial_value as second argument")
  local accum,key,value = initial_value
  for key,value in ... do
    accum = func(accum, value or key)
  end
  return accum
end

function apply(func, f, s, v)
  if not func then func = function() end end
  local tmp = table.pack(f(s,v))
  while tmp[1] ~= nil do
    func(table.unpack(tmp))
    tmp = table.pack(f(s,tmp[1]))
  end
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

function parallel_foreach(num_processes, list, func, output_serialization_function)
  local outputs
  if output_serialization_function then
    outputs = map(function(idx) return os.tmpname() end,
		  range(1,num_processes))
  end
  local id = util.split_process(num_processes)-1
  if outputs then
    local f = io.open(outputs[id+1], "w")
    fprintf(f, "return {\n")
    for index, value in ipairs(list) do
      if (index%num_processes) == id then
	local ret = func(value)
	fprintf(f,"[%d] = %s,\n",index,
		output_serialization_function(ret) or "nil")
      end
    end
    fprintf(f, "}\n")
    f:close()
    if id ~= 0 then os.exit(0) end
    util.wait()
    -- maps all the outputs to a table
    return map(function(v)return v end,
	       iterable_map(function(index,filename)
			      local t = dofile(filename)
			      os.remove(filename)
			      -- multiple outputs from this filename
			      apply(coroutine.yield, pairs(t))
			    end,
			    -- iterate over each output filename
			    ipairs(outputs)))
  else
    for index, value in ipairs(list) do
      if (index%num_processes) == id then
	local ret = func(value)
      end
    end
    if id ~= 0 then os.exit(0) end
    util.wait()
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

function range(...)
  local arg = table.pack(...)
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
  local params = params or {}
  local t      = t or {}
  local ret    = {}
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
      error("Incorrect type '" .. type(v) .. "' for field '" .. key .. "'")
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
  local arg = table.pack(...)
  return function(t)
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

-- auxiliary function for fast development of reductions
function math.add(a,b)
  if a and b then return a+b end
  if not a and not b then return function(a,b) return a+b end end
  if b == nil then
    return function(b) return a+b end
  end
end
-- auxiliary function for fast development of reductions
function math.sub(a,b)
  if a and b then return a-b end
  if not a and not b then return function(a,b) return a-b end end
  if b == nil then
    return function(b) return a-b end
  elseif a == nil then
    return function(a) return a-b end
  end
end
-- auxiliary function for fast development of reductions
function math.mul(a,b)
  if a and b then return a*b end
  if not a and not b then return function(a,b) return a*b end end
  if b == nil then
    return function(b) return a*b end
  end
end
-- auxiliary function for fast development of reductions
function math.div(a,b)
  if a and b then return a/b end
  if not a and not b then return function(a,b) return a/b end end
  if b == nil then
    return function(b) return a/b end
  elseif a == nil then
    return function(a) return a/b end
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

function math.median(t, ini, fin)
  local ini,fin = ini or 1, fin or #t
  local len     = fin-ini+1
  local mpos    = math.floor((ini+fin-1)/2)
  local median  = t[mpos]
  if len % 2 ~= 0 then
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

---------------------------------------------------------------
------------------------ STRING UTILS -------------------------
---------------------------------------------------------------

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

function table.tostring(t)
  local out = {}
  for i,v in pairs(t) do
    local key
    local value
    if tonumber(i) then key = "["..i.."]".."="
    else key = string.format("[%q]=",i)
    end
    if luatype(v) == "table" then value = "\n"..table.tostring(v)
    elseif luatype(v) == "string" then value = string.format("%q",v)
    else value = tostring(v)
    end
    table.insert(out, key .. value)
  end
  return "{\n"..table.concat(out,",").."\n}"
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

---------------------------------------------------------------
---------------------- ITERATOR CLASS -------------------------
---------------------------------------------------------------

-- The iterator class is useful to simplify the syntax of map, filter, reduce,
-- and apply functions, introducing a more natural order of application of the
-- functions.

local iterator_methods,
iterator_class_metatable = class("iterator")

function iterator_class_metatable:__call(...)
  local obj = { data=table.pack(...) }
  return class_instance(obj, iterator, true)
end

function iterator.meta_instance:__call() return table.unpack(self.data) end

function iterator_methods:get() return table.unpack(self.data) end

function iterator_methods:map(func)
  return iterator(iterable_map(func, self:get()))
end

function iterator_methods:filter(func)
  return iterator(iterable_filter(func, self:get()))
end

function iterator_methods:apply(func)
  apply(func, self:get())
end

function iterator_methods:reduce(func, initial_value)
  return reduce(func, initial_value, self:get())
end

function iterator_methods:concat(sep1,sep2)
  local sep1,sep2 = sep1 or "",sep2 or sep1 or ""
  local t = {}
  self:apply(function(...)
	       local arg = table.pack(...)
	       table.insert(t, string.format("%s", table.concat(arg, sep1)))
	     end)
  return table.concat(t, sep2)
end

function iterator_methods:field(...)
  local f,s,v = self:get()
  local arg   = table.pack(...)
  return iterator(function(s)
		    local tmp = table.pack(f(s,v))
		    if tmp[1] == nil then return nil end
		    v = tmp[1]
		    local ret = { }
		    for i=1,#tmp do
		      for j=1,#arg do
			table.insert(ret, tmp[i][arg[j]])
		      end
		    end
		    return table.unpack(ret)
		  end,s)
end

function iterator_methods:select(...)
  local f,s,v = self:get()
  local arg   = table.pack(...)
  for i=1,#arg do arg[i]=tonumber(arg[i]) assert(arg[i],"select: expected a number") end
  return iterator(function(s)
		    local tmp = table.pack(f(s,v))
		    if tmp[1] == nil then return nil end
		    v = tmp[1]
		    local selected = {}
		    for i=1,#arg do selected[i] = tmp[arg[i]] end
		    return table.unpack(selected)
		  end,s)
end

function iterator_methods:table(func)
  local t = {}
  local func = func or function(idx) return idx end
  local idx = 1
  self:apply(function(...)
	       local arg = ...
	       if select("#",...) > 1 then arg = table.pack(...) end
	       t[func(idx)] = arg
	       idx = idx + 1
	     end)
  return t
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

april_set_doc("signal.register",
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

april_set_doc("signal.release",
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
