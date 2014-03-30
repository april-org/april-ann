get_table_from_dotted_string("utest",true)
--
local function write(level,format,...)
  if level>0 or util.stdout_is_a_terminal() then
    printf(format,...)
  end
end
--
local testn = 0
local check = function (self,func,error_msg)
  testn=testn+1
  if not func() then
    write(1, "Test %d: %sfail%s\n", testn,
          ansi.fg.bright_red, ansi.fg.default)
    if error_msg then
      if type(error_msg) == "function" then
        error_msg()
      else
        write(1, "%s\n",tostring(error_msg))
      end
    end
    return false
  else
    write(0, "Test %d: %sok%s\n", testn,
          ansi.fg.bright_green, ansi.fg.default)
    return true
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
setmetatable(utest.check,{ __call = check })
