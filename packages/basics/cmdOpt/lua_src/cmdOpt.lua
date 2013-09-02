-- See the test file for details
-- A cmdOpt object is created and two main methods can be used:
--  - Generate help message
--  - parse arguments, which returns string (error) or a table

-- There are short (1 letter) and long options
-- The following formats are accepted:
-- -c if c does not require arguments
-- -cdef if c,d,e and f does not require arguments
-- -cdefblah if c,d,e does not require arguments and f can accept arguments
-- --longoption --longoption=value
-- long options can use a prefix is there is non ambiguity
-- the void long option "--" can be used to stop option list

-- a lua class:
local cmdopt_methods,
cmdopt_class_metatable = class("cmdOpt")

function cmdopt_methods:add_option(option)
  -- option is a table with the following parameters:
  --  index_name -> name of option in result table
  --  description -> useful to generate help description
  --  filter -> nil or a function to filter the value
  --  action -> to execute with the option value
  --  short -> a single letter, without "-", several options are allowed
  --  long -> name of long option
  --  mode -> 'usergiven' 'always', default 'usergiven'
  --  argument -> 'yes','no','optional'
  --  default_value -> default value when argument ~= 'no'
  local opt = {}
  table.insert(self.options,opt) -- insert options in list
  opt.filter        = option.filter
  opt.action        = option.action
  opt.index_name    = option.index_name
  opt.description   = option.description
  opt.argument      = option.argument
  opt.argument_name = option.argument_name or "VALUE"
  opt.mode          = option.mode or 'usergiven'
  opt.default_value = option.default_value
  --
  local shorts = option.short
  if shorts then
    if type(shorts) == "string" then shorts = { shorts } end
    if type(shorts) ~= "table" then error("cmdOpt constructor") end
    for i,letter in ipairs(shorts) do
      if #letter ~= 1 or letter == '-' then 
	error("cmdOpt: bad short option '"..letter.."'") 
      end
      if self.short_options[letter] ~= nil then
	error("cmdOpt: short option '"..letter.."'already defined") 
      end
      self.short_options[letter] = opt
    end
    opt.short_options = shorts
  end
  --
  local longs = option.long
  if longs then
    if type(longs) == "string" then longs = { longs } end
    if type(longs) ~= "table" then error("cmdOpt constructor") end
    for i,str in ipairs(longs) do
      if #str == 0 then 
	error("cmdOpt: long option must have size >0") 
      end
      if self.long_options[str] ~= nil then
	error("cmdOpt: long option '"..str.."'already defined") 
      end
      self.long_options[str] = opt
    end
  end
  opt.long_options = longs
end

-- constructor
-- tbl is a table with the following arguments:
-- posix_mode -> true in order to stop
-- program_name -> a string
-- argument_description -> string, description of positional arguments
-- the "vector part" of tbl are just options to add with add_option
function cmdopt_class_metatable:__call(tbl)
  local obj = {
    options       = {}, -- lista de opciones
    short_options = {}, -- diccionario opciones cortas
    long_options  = {}, -- diccionario opciones largas
    posix_mode    = tbl.posix_mode or false,
    program_name  = tbl.program_name or arg[0],
    argument_description = tbl.argument_description or "positional_arguments",
    main_description = tbl.main_description or "",
    author        = tbl.author or "",
    copyright     = tbl.copyright or "",
    see_also      = tbl.see_also or "",
  }
  obj = class_instance(obj, self, true)
  for i,option in ipairs(tbl) do
    obj:add_option(option)
  end
  return obj
end

-- cool function to generate the help text
function cmdopt_methods:generate_help()
  local message = {}
  table.insert(message,"USAGE:\n\t"..self.program_name)
  for i,opt in ipairs(self.options) do
    local total = 0
    if opt.short_options then total = #opt.short_options end
    if opt.long_options  then total = total+#opt.long_options end
    local remains=total
    if (total>0) then
      local mandatory = (opt.mode == "always" and not opt.default_value)
      table.insert(message,
		   ((not mandatory and " [") or " ")..(total>1 and "(" or ""))
      for j,short in ipairs(opt.short_options or {}) do
	local str="-"..short
	if opt.argument == "yes" then 
	  str=str..opt.argument_name
	elseif opt.argument ~= "no" then 
	  str=str.."["..opt.argument_name.."]"
	end
	remains=remains-1
	if remains>0 then str=str.."|" end
	table.insert(message,str)
      end
      if opt.long_options then
	for j,long in ipairs(opt.long_options or {}) do
	  local str="--"..long
	  if opt.argument == "yes" then 
	    str=str.."="..opt.argument_name
	  elseif opt.argument ~= "no" then 
	    str=str.."[="..opt.argument_name.."]" 
	  end
	  remains=remains-1
	  if remains>0 then str=str.."|" end
	  table.insert(message,str)
	end
      end
      table.insert(message,(total>1 and ")" or "")..
		     ((not mandatory and "]") or " "))
    end
  end
  table.insert(message," ")
  table.insert(message,self.argument_description)
  table.insert(message,"\n\nDESCRIPTION:\n")
  if self.main_description ~= "" then
    table.insert(message,"\t"..self.main_description.."\n\n")
  end
  -- usage line is finished, now we proceed to describe every argument:
  for i,opt in ipairs(self.options) do
    local total = 0
    if opt.short_options then total = #opt.short_options end
    if opt.long_options  then total = total+#opt.long_options end
    local remains=total
    if (total>0) then
      local mandatory = (opt.mode == "always" and not opt.default_value)
      table.insert(message,"\t")
      for j,short in ipairs(opt.short_options or {}) do
	local str="-"..short
	if opt.argument == "yes" then 
	  str=str..opt.argument_name
	elseif opt.argument ~= "no" then 
	  str=str.."["..opt.argument_name.."]"
	end
	remains=remains-1
	if remains>0 then str=str..", " end
	table.insert(message,str)
      end
      if opt.long_options then
	for j,long in ipairs(opt.long_options or {}) do
	  local str="--"..long
	  if opt.argument == "yes" then 
	    str=str.."="..opt.argument_name
	  elseif opt.argument ~= "no" then 
	    str=str.."[="..opt.argument_name.."]" 
	  end
	  remains=remains-1
	  if remains>0 then str=str..", " end
	  table.insert(message,str)
	end
      end
      if opt.default_value == nil then
	table.insert(message,"\t      "..(opt.description or "")..
		       ((not mandatory and " "..ansi.fg["green"].."(optional)")
			  or "")..ansi.fg["default"].."\n")
      else
	table.insert(message,"\t      "..(opt.description or "")..
		       ((not mandatory and " "..ansi.fg["green"].."(optional)")
			  or "").." [DEFAULT: "..
		       tostring(opt.default_value) .. "]"..
		       ansi.fg["default"].."\n")
      end
    end
  end
  if self.author ~= "" then
    table.insert(message,"AUTHOR\n\t"..self.author.."\n\n")
  end
  if self.copyright ~= "" then
    table.insert(message,"COPYRIGHT\n\t"..self.copyright.."\n\n")
  end
  if self.see_also ~= "" then
    table.insert(message,"SEE ALSO\n\t"..self.see_also.."\n\n")
  end
  return table.concat(message)
end

-- auxiliary method
function cmdopt_methods:search_long_option(str)
  -- prefer exact match
  if self.long_options[str] then
    return self.long_options[str]
  end
  -- search for prefixes
  local strlen = #str
  local found
  for optname,opt in pairs(self.long_options) do
    if string.sub(optname,1,strlen) == str then
      if found ~= nil then -- hay conflicto, el prefijo es ambiguo
	return nil
      else
	found = opt
      end      
    end
  end
  return found
end

-- main method, returns a string in case of error (error message)
-- or a table with positinal and rest of arguments
-- actions are simply executed before return
function cmdopt_methods:parse_without_check(arguments)
  local arguments = arguments or arg -- arg is the global variable
  local opt_list = {} -- list of options to process
  local result   = {} -- result table
  local consider_options = true
  local pos=1
  local nargs=#arguments 
  local i=1
  while i<=nargs do
    local key,value,opt
    local str = arguments[i]
    if consider_options and string.sub(str, 1, 2) == "--" then -- long option
      local pos = string.find(str, "=", 1, true)
      if pos then
	key   = string.sub(str, 3, pos-1)
	value = string.sub(str, pos+1)
      else
	key   = string.sub(str, 3)
      end
      --printf("jarl key %s value %s",key,value or "nil")
      if key == "" then -- "--" is used to stop considering options
	consider_options = false
      else
	-- search long option (prefix) in table:
	opt = self:search_long_option(key)
	if opt == nil then -- error
	  return "ambiguous or non-existent long option: "..str
	end
	if opt.argument == "no" then
	  if value ~= nil then
	    return "long option '"..key.."' has no arguments but received argument "..value
	  else
	    value = true
	  end
	end
	if opt.argument == "yes" and value == nil then
	  return "long option '"..key.."' has obligatory argument"
	end
	-- guardamos opt y value para procesar
	table.insert(opt_list,{opt,value})      
      end
    elseif consider_options and string.sub( str, 1, 1 ) == "-" then -- short option
      -- vamos a procesar las opciones de izquierda a derecha
      -- mientras encontremos opciones sin argumentos
      -- si encontramos una opcion con argumento opcional u
      -- obligatorio procesamos el resto de la cadena como su valor, a
      -- menos que sea el ultimo caracter, en cuyo caso tomamos el
      -- argumento siguiente como su valor solamente en caso de tener
      -- argumento obligatorio
      local j=2
      local strlen = #str
      local go_on = true
      while j<=strlen and go_on do
	key = string.sub(str, j, j)
	opt = self.short_options[key]
	if opt == nil then -- error
	  return "short option '"..key.."' not found"
	end
	if opt.argument == 'no' then
	  -- store opt and value to process later
	  value = true
	else -- argument is obligatory or optional
	  go_on = false -- stop
	  -- let's see the rest of the string
	  value = string.sub( str, j+1 )
	  if value == "" and opt.argument == 'yes' then
	    i=i+1
	    if i>nargs then -- error
	      return "short option '"..key.."' expects obligatory argument"
	    end
	    value = arguments[i]
	  end
	end
	table.insert(opt_list,{opt,value})
	j=j+1
      end -- while that traverses string
    else -- if else that check "-" and "--"
      table.insert(result,str) -- positional argument
      if self.posix_mode then consider_options = false end
    end
    i = i+1
  end -- while i<nargs do
  -- now process list of options
  for i,optvalue in ipairs(opt_list) do
    local opt   = optvalue[1]
    local value = optvalue[2]
    value = value or opt.default_value
    printf("# ARG %s = %s\n", opt.index_name or "nil", tostring(value or "nil"))
    if type(opt.filter) == "function" then value = opt.filter(value) end
    if opt.index_name then result[opt.index_name] = value end
    if type(opt.action) == "function" then opt.action(value) end
  end
  return result
end

function cmdopt_methods:check_args(optargs,initial_values)
  local initial_values = initial_values or {}
  for _,opt in pairs(self.options) do
    local idx = opt.index_name
    if idx then
      if not optargs[idx] and initial_values[idx] then
	local value = initial_values[idx]
        printf("# INITIAL %s = %s\n", idx, tostring(value or "nil"))
        if type(opt.filter) == "function" then value = opt.filter(value) end
	if type(opt.action) == "function" then opt.action(value) end
	optargs[idx] = value
      elseif opt.mode == 'always' and optargs[idx] == nil then
	local value = opt.default_value
        printf("# DEFAULT %s = %s\n", idx, tostring(value or "nil"))
	assert(value ~= nil,
	       (table.concat(opt.short_options or opt.long_options)..
		  " option is mandatory!!"))
	if type(opt.filter) == "function" then value = opt.filter(value) end
	if type(opt.action) == "function" then opt.action(value) end
	optargs[idx] = value
      end
    end
  end
  return optargs
end

function cmdopt_methods:parse_args(arguments)
  local result = self:parse_without_check(arguments)
  self:check_args(result)
  return result
end
