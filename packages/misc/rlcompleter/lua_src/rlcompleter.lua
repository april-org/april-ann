-- Lua side of readline completion for REPL
-- By Patrick Rapin; adapted by Reuben Thomas
-- Adapted to April-ANN by Francisco Zamora-Martinez, 2013

-- The list of Lua keywords
local keywords = {
  'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
  'function', 'if', 'in', 'local', 'nil', 'not', 'or', 'repeat',
  'return', 'then', 'true', 'until', 'while'
}

-- This function is called back by C function do_completion, itself called
-- back by readline library, in order to complete the current input line.
rlcompleter._set(
  function (word, line, startpos, endpos)
    -- Helper function registering possible completion words, verifying matches.
    local matches = {}
    local function add(value)
      value = tostring(value)
      if value:match("^" .. word) then
        matches[#matches + 1] = value
      end
    end
    
    -- This function does the same job as the default completion of readline,
    -- completing paths and filenames. Rewritten because
    -- rl_basic_word_break_characters is different.
    -- Uses LuaFileSystem (lfs) module for this task.
    local function filename_list(str)
      local path, name = str:match("(.*)[\\/]+(.*)")
      path = (path or ".") .. "/"
      name = name or str
      local d = rlcompleter.dir(path)
      if d then
	for f in d:iterate() do
	  if rlcompleter.dir.isdir(path .. f) then
	    add(f .. "/")
	  else
	    add(f)
	  end
	end
      else
	add("")
      end
    end

    -- This function is called in a context where a keyword or a global
    -- variable can be inserted. Local variables cannot be listed!
    local function add_globals()
      for _, k in ipairs(keywords) do
        add(k)
      end
      for k in pairs(_G) do
        add(k)
      end
    end

    -- Main completion function. It evaluates the current sub-expression
    -- to determine its type. Currently supports tables fields, global
    -- variables and function prototype completion.
    local function contextual_list(expr, sep, str)
      if str then
        return filename_list(str)
      end
      if expr and expr ~= "" then
        local v = load("return " .. expr)
        if v then
          v = v()
          local t = luatype(v)
          if sep == '.' or sep == ':' then
            if t == 'table' then
              for k, v in pairs(v) do
                if luatype(k) == 'string' and (sep ~= ':' or luatype(v) == "function") then
                  add(k)
                end
              end
            end
	    if getmetatable(v) then
	      local aux = v
	      repeat
		local mt = getmetatable(aux)
		if mt.__index and luatype(mt.__index) == 'table' then
		  for k,v in pairs(mt.__index) do
		    add(k)
		  end
		end
		aux = mt.__index
	      until not aux or not getmetatable(aux)
	    end
          elseif sep == '[' then
            if t == 'table' then
              for k in pairs(v) do
                if luatype(k) == 'number' then
                  add(k .. "]")
                end
              end
              if word ~= "" then add_globals() end
            end
          end
        end
      end
      if #matches == 0 then
        add_globals()
      end
    end

    -- This complex function tries to simplify the input line, by removing
    -- literal strings, full table constructors and balanced groups of
    -- parentheses. Returns the sub-expression preceding the word, the
    -- separator item ( '.', ':', '[', '(' ) and the current string in case
    -- of an unfinished string literal.
    local function simplify_expression(expr)
      -- Replace annoying sequences \' and \" inside literal strings
      expr = expr:gsub("\\(['\"])", function (c)
                                      return string.format("\\%03d", string.byte(c))
                                  end)
      local curstring
      -- Remove (finished and unfinished) literal strings
      while true do
        local idx1, _, equals = expr:find("%[(=*)%[")
        local idx2, _, sign = expr:find("(['\"])")
        if idx1 == nil and idx2 == nil then
          break
        end
        local idx, startpat, endpat
        if (idx1 or math.huge) < (idx2 or math.huge) then
          idx, startpat, endpat = idx1, "%[" .. equals .. "%[", "%]" .. equals .. "%]"
        else
          idx, startpat, endpat = idx2, sign, sign
        end
        if expr:sub(idx):find("^" .. startpat .. ".-" .. endpat) then
          expr = expr:gsub(startpat .. "(.-)" .. endpat, " STRING ")
        else
          expr = expr:gsub(startpat .. "(.*)", function (str)
                                                 curstring = str
                                                 return "(CURSTRING "
                                             end)
        end
      end
      expr = expr:gsub("%b()"," PAREN ") -- Remove groups of parentheses
      expr = expr:gsub("%b{}"," TABLE ") -- Remove table constructors
      -- Avoid two consecutive words without operator
      expr = expr:gsub("(%w)%s+(%w)","%1|%2")
      expr = expr:gsub("%s+", "") -- Remove now useless spaces
      -- This main regular expression looks for table indexes and function calls.
      return curstring, expr:match("([%.%w%[%]_]-)([:%.%[%(])" .. word .. "$")
    end
    -- Now call the processing functions and return the list of results.
    local str, expr, sep = simplify_expression(line:sub(1, endpos))
    
    contextual_list(expr, sep, str)
    
    if #matches == 1 and word == matches[1] then
      print("\n----------------- DOCUMENTATION ----------------------")
      local m  = matches[1]
      local id = expr or ""
      local mt
      if expr and #expr > 0 then
	local v = load("return " .. expr)
	if v then	
	  v = v()
	  if v then mt = getmetatable(v) end
	  if mt and mt.id then id = mt.id:match("^[^%s]+") end
	end
      end
      local prefix = id
      if #prefix > 0 then prefix = prefix .. "." end
      april_print_doc(prefix .. m, 2)
      local v = load("return " .. prefix .. m)
      if v then
	v = v()
	if v then mt = getmetatable(v) end
	if mt and mt.__call then
	  april_print_doc(prefix .. m .. ".__call", 2)
	end
      end
      print("------------------------------------------------------")
    end
    
    return matches
  end
)
