-- Lua side of readline completion for REPL
-- By Patrick Rapin; adapted by Reuben Thomas
-- Adapted to APRIL-ANN by Francisco Zamora-Martinez, 2013

-- escape all characters to avoid problems with Lua regular expressions
local function escape(str)
  return str:gsub("(.)","%%%1")
end

-- Returns index_table field from a metatable, which is a copy of __index table.
local function get_index(mt)
  return mt.index_table
end

-- The list of Lua keywords
local keywords = {
  'and', 'break', 'do', 'else', 'elseif', 'end', 'false', 'for',
  'function', 'if', 'in', 'local', 'nil', 'not', 'or', 'repeat',
  'return', 'then', 'true', 'until', 'while'
}

-- in case you want to use this software without APRIL-ANN
local luatype = luatype or type

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
        return
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
                local idx = get_index(mt)
		if idx and luatype(idx) == 'table' then
		  for k,v in pairs(idx) do
		    add(k)
		  end
		end
                if rawequal(aux,idx) then break end -- avoid infinite loops
		aux = idx
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
      if #matches == 0 then add_globals() end
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
          -- equals = escape(equals)
          idx, startpat, endpat = idx1, "%[" .. equals .. "%[", "%]" .. equals .. "%]"
        else
          -- sign = escape(sign)
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
    
    return matches
  end
)
