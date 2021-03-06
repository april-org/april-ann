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
local cmdOpt,cmdopt_methods = class("cmdOpt")
_G.cmdOpt = cmdOpt

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
  local opt = get_table_fields(
    {
      filter = { type_match="function", default=nil },
      action = { type_match="function", default=nil },
      index_name = { type_match="string" },
      description = { mandatory=true, type_match="string" },
      argument = { mandatory=true, type_match="string" },
      argument_name = { type_match="string", default="VALUE" },
      mode = { type_match="string", default="usergiven" },
      default_value = { default=nil },
      short = { default=nil },
      long = { default=nil },
    }, option)
  assert(opt.argument=="yes" or opt.argument=="no" or opt.argument=="optional" or opt.argument==nil,
	 "Expected 'yes', 'no', 'optional' or nil in argument field")
  assert(opt.short or opt.long,
	 "It is mandatory to give a long or a short field")
  opt.short,opt.long=nil,nil
  table.insert(self.options,opt) -- insert options in list
  --
  assert(opt.mode=="always" or opt.mode=="usergiven",
	 "Given incorrect mode '"..tostring(opt.mode).."'. Use 'usegiven' or 'always'")
  --
  if opt.default_value and not opt.mode=="always" then
    error("default_value only allowed if mode='always', otherwise is forbidden")
  end
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
function cmdOpt:constructor(tbl)
  self.options       = {} -- lista de opciones
  self.short_options = {} -- diccionario opciones cortas
  self.long_options  = {} -- diccionario opciones largas
  self.posix_mode    = tbl.posix_mode or false
  self.program_name  = tbl.program_name or arg[0]
  self.argument_description = tbl.argument_description or "positional_arguments"
  self.main_description = tbl.main_description or ""
  self.author        = tbl.author or ""
  self.copyright     = tbl.copyright or ""
  self.see_also      = tbl.see_also or ""
  for i,option in ipairs(tbl) do
    self:add_option(option)
  end
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

-- main method, throws an error in case of fail, or returns
-- or a table with positinal and rest of arguments
-- actions are simply executed before return
function cmdopt_methods:parse_without_check(arguments, verbose)
  if verbose == nil then verbose = true end
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
        assert(opt,"ambiguous or non-existent long option: "..str)
	if opt.argument == "no" then
          april_assert(value == nil,
                       "long option '%s' has no arguments but received argument %s",
                       key,value)
          value = true
	end
        assert(opt.argument ~= "yes" or value ~= nil,
               "long option '"..key.."' has obligatory argument")
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
        assert(opt, "short option '"..key.."' not found")
	if opt.argument == 'no' then
	  -- store opt and value to process later
	  value = true
	else -- argument is obligatory or optional
	  go_on = false -- stop
	  -- let's see the rest of the string
	  value = string.sub( str, j+1 )
	  if value == "" and opt.argument == 'yes' then
	    i=i+1
            assert(i<=nargs,"short option '"..key.."' expects obligatory argument")
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
    if verbose then printf("# ARG %s = %s\n", opt.index_name or "nil", tostring(value or "nil")) end
    if type(opt.filter) == "function" then
      value = opt.filter(value)
      april_assert(value ~= nil, "Filter fails for option %s", opt.index_name)
    end
    if opt.index_name then result[opt.index_name] = value end
    if type(opt.action) == "function" then opt.action(value) end
  end
  return result
end

function cmdopt_methods:check_args(optargs,initial_values,verbose)
  if verbose==nil then verbose=true end
  local initial_values = initial_values or {}
  for _,opt in pairs(self.options) do
    local idx = opt.index_name
    if idx then
      if not optargs[idx] and initial_values[idx] then
	local value = initial_values[idx]
        if verbose then printf("# INITIAL %s = %s\n", idx, tostring(value or "nil")) end
	if type(opt.filter) == "function" then
          value = opt.filter(value)
          april_assert(value ~= nil, "Filter fails for option %s", idx)
        end
	if type(opt.action) == "function" then opt.action(value) end
	optargs[idx] = value
      elseif opt.mode == 'always' and optargs[idx] == nil then
	local value = opt.default_value
        if verbose then printf("# DEFAULT %s = %s\n", idx, tostring(value or "nil")) end
	assert(value ~= nil,
	       (table.concat(opt.short_options or opt.long_options)..
		  " option is mandatory!!"))
	if type(opt.filter) == "function" then
          value = opt.filter(value)
          april_assert(value ~= nil, "Filter fails for option %s", idx)
        end
	if type(opt.action) == "function" then opt.action(value) end
	optargs[idx] = value
      end
    end
  end
  return optargs
end

function cmdopt_methods:parse_args(arguments, defopt, verbose)
  local defopt_func = defopt_func or function(v) return v end
  local result = self:parse_without_check(arguments, verbose)
  local initial_values
  if defopt and result[defopt] then
    initial_values = result[defopt]
    result[defopt] = nil
  end
  self:check_args(result, initial_values, verbose)
  return result
end
