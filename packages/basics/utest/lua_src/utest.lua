get_table_from_dotted_string("utest",true)
--
local function write(level,format,...)
  if level>0 or util.stdout_is_a_terminal() then
    printf(format,...)
  end
end
--
local testn  = 0
local passed = 0
local failed = 0
utest.finish = function()
  write((failed == 0 and 0) or 1, "%d total tests: %s%d passed%s, %s%d failed%s\n",
        testn,
        ansi.fg.bright_green, passed, ansi.fg.default,
        ansi.fg.bright_red, failed, ansi.fg.default)
  assert(failed == 0, "%sTest failed%s"%{ansi.fg.bright_red, ansi.fg.default})
  write(0, "%sOk%s\n", ansi.fg.bright_green, ansi.fg.default)
end
local function register()
  -- global
  utest.__initialized__ = {}
  setmetatable(utest.__initialized__, { __gc = utest.finish })
end
--
local check = function (func,error_msg)
  if not utest.__initialized__ then register() end
  testn = testn + 1
  local ret = table.pack(xpcall(func,debug.traceback))
  local ok  = table.remove(ret,1)
  local success = ret[1]
  if ok and success then
    write(0, "Test %d: %sok%s\n", testn,
          ansi.fg.bright_green, ansi.fg.default)
    passed = passed + 1
    return true
  else
    if not ok then
      debug.traceback()
      -- protected call to write, in case tostring fails with returned values
      pcall(function()
              write(1, "%s\n",
                    iterator(ipairs(ret)):select(2):map(tostring):concat(" "))
            end)
    end
    write(1, "Test %d: %sfail%s\n", testn,
          ansi.fg.bright_red, ansi.fg.default)
    if error_msg then
      if type(error_msg) == "function" then
        error_msg()
      else
        write(1, "%s\n",tostring(error_msg))
      end
    end
    failed = failed + 1
    return false
  end
end
utest.check = { }
--
utest.check.eq = function(a, b, ...)
  return check(function() return a == b end, ...)
end
utest.check.neq = function(a, b, ...)
  return check(function() return a ~= b end, ...)
end
utest.check.lt = function(a, b, ...)
  return check(function() return a < b end, ...)
end
utest.check.le = function(a, b, ...)
  return check(function() return a <= b end, ...)
end
utest.check.gt = function(a, b, ...)
  return check(function() return a > b end, ...)
end
utest.check.ge = function(a, b, ...)
  return check(function() return a > b end, ...)
end
--
setmetatable(utest.check,{ __call = function(self,...) return check(...) end })
