-- This module allows unit testing in APRIL-ANN. This module uses internal
-- variables to account for 'passed' and 'failed' tests. At the end of the
-- program a summary will be printed to stderr. In case of a non-terminal
-- output, only the 'failed' tests will produce a print out. This module exports
-- the following functions:
--
-- utest.check(function, error_message): executes the given function, in case of
--                                       fail, it shows the error_message
--
-- utest.check.success(function, error_message): an alias of previous one
--
-- utest.check.fail(function, error_message): the opposite of previous one
--
-- utest.check.number_eq(a, b, epsilon, error_message): tests of a == b with epsilon tolerance
--
-- utest.check.eq(a, b, error_message): tests of a == b
--
-- utest.check.neq(a, b, error_message): tests of a ~= b
--
-- utest.check.lt(a, b, error_message): tests of a < b
--
-- utest.check.le(a, b, error_message): tests of a <= b
--
-- utest.check.gt(a, b, error_message): tests of a > b
--
-- utest.check.ge(a, b, error_message): tests of a >= b
--
-- utest.check.TRUE(a, error_message): tests of a == true
--
-- utest.check.FALSE(a, error_message): tests of a == false
-------------------------------------------------------------------------------
utest = utest or {}
--
local function write(level,format,...)
  if level>0 or util.stdout_is_a_terminal() then
    fprintf(io.stderr, format,...)
  end
end
--
utest.warning = function(format,...)
  return fprintf(io.stderr,
                 table.concat({
                     "\t",
                     ansi.fg.bright_yellow,
                     "Warning: ",
                     ansi.fg.default,
                     format
                 }),
                 ...)
end
--
local NONAMED = "UNKNOWN"
local test_name = NONAMED
local testn  = 0
local passed = 0
local failed = 0
local failed_list = {}
local names_order = {}
utest.finish = function()
  write((failed > 0 and 1) or 0, "%d total tests: %s%d passed%s, %s%d failed%s\n",
        testn,
        ansi.fg.bright_green, passed, ansi.fg.default,
        ansi.fg.bright_red, failed, ansi.fg.default)
  if failed > 0 then
    for _,name in ipairs(names_order) do
      write(1, "%sTest failed%s %s\tList: [ %s ]\n"%
              {ansi.fg.bright_red,
               ansi.fg.default,
               name,
               #failed_list[name] < 20 and table.concat( failed_list[name], ", ") or "too large to be displayed"})
    end
    os.exit(1)
  end
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
    write(0, "Test %s %d: %sok%s\n", test_name, testn,
          ansi.fg.bright_green, ansi.fg.default)
    passed = passed + 1
    return true
  else
    if not ok then
      -- protected call to write, in case tostring fails with returned values
      pcall(function()
              write(1, "%s\n",
                    iterator(ipairs(ret)):select(2):map(tostring):concat(" "))
            end)
      write(1, "%s\n", debug.traceback())
    end
    write(1, "Test %s %d: %sfail%s", test_name, testn,
          ansi.fg.bright_red, ansi.fg.default)
    if error_msg then
      if type(error_msg) == "function" then
        error_msg = error_msg() or ""
      else
        error_msg = tostring(error_msg)
      end
      error_msg = ", msg: %s"%{error_msg}
    end
    write(1, "%s\n", error_msg or "")
    failed = failed + 1
    if not failed_list[test_name] then
      table.insert(names_order, test_name)
      failed_list[test_name] = {}
    end
    table.insert(failed_list[test_name], testn)
    return false
  end
end
utest.check = { }
--
utest.check.number_eq = function(a, b, epsilon, ...)
  local epsilon = epsilon or 0.02
  return check(function()
                 return math.abs(a-b)/math.abs(a+b) < epsilon
               end, ...)
end
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
utest.check.success = check
utest.check.fail = function(f, ...)
  return check(function() return not f() end, ...)
end
utest.check.TRUE = function(a, ...)
  return check(function() return a end, ...)
end
utest.check.FALSE = function(a, ...)
  return check(function() return not a end, ...)
end
--_
utest.test = function(name, test_func)
  assert( test_name == NONAMED )
  assert( type(name) == "string", "Needs a string as first argument" )
  assert( type(test_func) == "function", "Needs a function as second argument")
  test_name = name
  test_func()
  test_name = NONAMED
end
--
setmetatable(utest.check,{ __call = function(self,...) return check(...) end })
