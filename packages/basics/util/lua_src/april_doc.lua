local COLWIDTH=70
local DOC_TABLE = setmetatable({}, {__mode = "k" })

-- help documentation
local allowed_classes = {
  ["class"]=true,
  ["namespace"]=true,
  ["function"]=true,
  ["method"]=true,
  ["var"]=true
}
function april_set_doc(object, docblock)
  local function concat(aux)
    local taux = luatype(aux)
    if taux == "table" then return table.concat(aux, " ") end
    april_assert(taux == "string", "Expected a string, found %s", taux)
    return aux
  end
  --
  local docblock = get_table_fields(
    {
      class       = { mandatory=true,  type_match="string", default=nil },
      summary     = { mandatory=true },
      description = { mandatory=false, default=docblock.summary },
      params      = { mandatory=false, default=nil },
      outputs     = { mandatory=false, default=nil },
    }, docblock)
  assert(allowed_classes[docblock.class], "Incorrect class: " .. docblock.class)
  docblock.summary     = concat(docblock.summary)
  docblock.description = concat(docblock.description)
  if type(docblock.params) == "string" then docblock.params = { docblock.params } end
  if type(docblock.outputs) == "string" then docblock.outputs = { docblock.outputs } end
  assert(not docblock.params or type(docblock.params) == "table",
         "Params filed needs to be nil, table or string")
  assert(not docblock.outputs or type(docblock.outputs) == "table",
         "Outputs filed needs to be nil, table or string")
  --
  local tt = type(object)
  if not object or tt == "string" or tt == "number" or tt == "boolean" then
    fprintf(io.stderr, "Error in april_set_doc\n class: %s\n summary: %s\n",
            docblock.class, docblock.summary)
    assert(tt ~= "string", "Unable to add string doc")
    assert(tt ~= "boolean", "Unable to add boolean doc")
    assert(tt ~= "number", "Unable to add number doc")
    assert(object, "Needs a non-nil value as first argument")
  end
  --
  if docblock.params then
    for i,v in pairs(docblock.params) do docblock.params[i] = concat(v) end
  end
  if docblock.outputs then
    for i,v in pairs(docblock.outputs) do docblock.outputs[i] = concat(v) end
  end
  DOC_TABLE[object] = DOC_TABLE[object] or {}
  table.insert(DOC_TABLE[object], docblock)
  return DOC_TABLE[object]
end

function april_print_doc(object, verbosity, prefix)
  assert(object ~= nil, "Needs any object as first argument")
  assert(type(verbosity)=="number",  "Needs a number as second argument")
  local prefix = prefix or ""
  local current_table = DOC_TABLE[object]
  if not current_table then
    if #prefix > 0 then print(prefix) end
    if verbosity > 1 then
      print("No documentation found.")
    end
    return
  end
  local function build_short(str, color1)
    if #prefix > 0 then
      return { prefix,
               ansi.fg[color1]..str or "",
               ansi.fg["default"] }
    else
      return { ansi.fg[color1]..str or "",
               ansi.fg["default"] }
    end
  end
  local function build_list(out, list, desc)
    if list then
      table.insert(out,
                   { "\n"..ansi.fg["cyan"]..desc..ansi.fg["default"] })
      local names_table = {}
      for name,_ in pairs(list) do table.insert(names_table,name) end
      table.sort(names_table, function(a,b) return tostring(a)<tostring(b) end)
      for k,name in ipairs(names_table) do
        local description = list[name]
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
    return out
  end

  for idx,current in ipairs(current_table) do

    if idx > 1 and verbosity > 1 then
      print("--------------------------------------------------------------\n")
    end
    --local name = current.name
    local out = {}
    table.insert(out, build_short(current.class, "bright_red"))
    if verbosity > 0 then
      if current.summary then
        local aux = "          "
        local str = string.truncate(current.summary, COLWIDTH,
                                    aux..aux..aux)
        str = string.gsub(str, "%[(.*)%]",
                          "["..ansi.fg["bright_yellow"].."%1"..
                            ansi.fg["default"].."]")
        table.insert(out[#out], str)
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
      build_list(out, current.params, "parameters:")
      build_list(out, current.outputs, "outputs:")
    end
    for i=1,#out do out[i] = table.concat(out[i], " ") end
    print(table.concat(out, "\n"))
    if verbosity > 1 then print("") end
  end
end

-- verbosity => 0 only names, 1 only summary, 2 all
function april_help(object, verbosity)
  local verbosity = verbosity or 2
  if verbosity > 0 then
    if ( luatype(object) == "table" and
         object.meta_instance and object.meta_instance.id ) then
      print(ansi.fg["cyan"].."ID: "..ansi.fg["default"]..object.meta_instance.id)
    elseif get_object_id(object) then
      print(ansi.fg["cyan"].."ID: "..ansi.fg["default"]..get_object_id(object))
    end
  end
  local object = object or _G
  ----------------------------------------------------------------------------
  -- AUXILIARY FUNCTIONS
  local function print_result(aux)
    local prev = { }
    table.sort(aux, function(a,b) return a[1]<b[1] end)
    for i,v in ipairs(aux) do
      if v[1] ~= prev[1] then
        april_print_doc(v[3], math.min(1, verbosity),
                        ansi.fg["cyan"].."   * "..
                          v[2]..ansi.fg["default"].." "..v[1])
      end
      prev = v
      -- print_data(v)
    end
    print("")
  end
  local dummy_filter_function = function() return true end
  local function process_pairs(title, tbls, filter)
    local filter = filter or dummy_filter_function
    local aux = {}
    for _,t in ipairs(tbls) do
      for i,v in pairs(t) do
        if filter(i,v) then
          table.insert(aux, {i,"",v})
        end
      end
    end
    if #aux > 0 then
      print(ansi.fg["cyan"].." -- "..title..ansi.fg["default"])
      print_result(aux)
    end
  end
  local function print_inheritance(title, object)
    while ( getmetatable(object) and getmetatable(object).__index and
            not rawequal(getmetatable(object).__index, object) ) do
      local mt = getmetatable(object)
      local superclass_name = (mt.id and mt.id:gsub(" class","")) or "UNKNOWN"
      object = mt.__index
      process_pairs(title..superclass_name, { object },
                    function(k,v) return luatype(v) == "function" end)
    end
  end
  ----------------------------------------------------------------------------
  -- documentation
  april_print_doc(object, verbosity)
  if class.is_class(object) and DOC_TABLE[object.constructor] then
    print("--------------------------------------------------------------\n")
    -- constructor documentation is defined at class_table.constructor method
    april_print_doc(object.constructor, verbosity)
  end
  local mt = getmetatable(object)
  if mt then
    -- metatable constructor and destructor
    if mt.__call then
      if DOC_TABLE[mt.__call] then
        -- constructor documentation is defined at
        -- getmetatable(class_table).__call metamethod
        print("--------------------------------------------------------------\n")
        april_print_doc(mt.__call, verbosity)
      end
    end
    -- metatable constructor and destructor
    print("--------------------------------------------------------------\n")
    process_pairs("metatable", { mt, mt.__index },
                  function(i,v) return luatype(v) == "function" end)
    if mt.__index then
      print_inheritance("inherited metatable from ", mt.__index)
    end
  end
  if luatype(object) == "table" then
    -- OBJECT class content
    print("--------------------------------------------------------------\n")
    -- local print_data = function(d) print("\t * " .. d) end
    local classes    = {}
    local funcs      = {}
    local names      = {}
    local vars       = {}
    for i,v in pairs(object) do
      if i ~= "meta_instance" then
        if class.is_class(v) then
          table.insert(classes, {i, "", v})
        elseif iscallable(v) then
          table.insert(funcs, {i, string.format("%8s",luatype(v)), v})
        elseif luatype(v) == "table" and object.meta_instance then
          table.insert(names, {i, "", v})
        else
          table.insert(vars, {i, string.format("%8s",luatype(v)), v})
        end
      end
    end
    if #vars > 0 then
      print(ansi.fg["cyan"].." -- basic variables (string, number)"..
              ansi.fg["default"])
      print_result(vars)
    end
    if #names > 0 then
      print(ansi.fg["cyan"].." -- names in the namespace"..ansi.fg["default"])
      print_result(names)
    end
    if #classes > 0 then
      print(ansi.fg["cyan"].." -- classes in the namespace"..ansi.fg["default"])
      print_result(classes)
    end
    if #funcs > 0 then
      print(ansi.fg["cyan"].." -- static functions or tables"..ansi.fg["default"])
      print_result(funcs)
    end
    -- OBJECT meta_instance
    if object.meta_instance and object.meta_instance.__index then
      process_pairs("object metatable", { object.meta_instance })
      process_pairs("object methods", { object.meta_instance.__index },
                    function(i,v) return luatype(v) == "function" end)
      --
      print_inheritance("inherited methods from ", object.meta_instance.__index)
    end
  end
  print()
end
  
function april_dir(t, verbosity)
  april_help(t, 0)
end

local april_doc_mt = {
  __concat = function(a,b)
    local tt = luatype(b)
    if tt == "table" and april_doc_mt == getmetatable(b) then
      table.insert(a,b[1])
      return a
    else
      for _,t in ipairs(a) do
        april_set_doc(b,t)
      end
      return b
    end
  end,
}
function april_doc(t)
  return setmetatable({t}, april_doc_mt)
end
