get_table_from_dotted_string("utest",true)
--
local function write(level,format,...)
  if level>0 or util.stdout_is_a_terminal() then
    printf(format,...)
  end
end
--
local testn = 1
function utest.check(func,msg)
  if not func() then
    write(1, "Test %d: %sfail%s %s\n", testn,
          ansi.fg.bright_red, ansi.fg.default, msg or "")
  else
    write(0, "Test %d: %sok%s %s\n", testn,
          ansi.fg.bright_green, ansi.fg.default, msg or "")
  end
  testn=testn+1
end
