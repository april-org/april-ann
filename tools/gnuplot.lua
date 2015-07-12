local gnuplot = {} -- module gnuplot

--------------- GNUPLOT OBJECT METHODS ---------------------
local gnuplot_methods = {}

-----------------------
-- PRIVATE FUNCTIONS --
-----------------------

-- Writes using format and a list of arguments
local function writef(self,format, ...)
  self.in_pipe:write(string.format(format,...))
  return self
end

-- Writes the given strings (separated by blanks)
local function write(self,...)
  self.in_pipe:write(table.concat(table.pack(...), " "))
  return self
end

local function read(output,value,format)
  local format = format or "%s"
  if not value then return "" end
  return string.format("%s "..format, output, tostring(value))
end

--------------------
-- PUBLIC METHODS --
--------------------

-- Writes a line using format and a list of arguments
function gnuplot_methods:writeln(format, ...)
  writef(self,format,...)
  write(self,"\n")
  self:flush()
  return self
end

-- Sets a parameter
function gnuplot_methods:set(...)
  write(self,"set ")
  write(self,table.concat(table.pack(...), " "))
  write(self,"\n")
  self:flush()
  return self
end

-- Unsets a parameter
function gnuplot_methods:unset(...)
  write(self,"unset ")
  write(self,table.concat(table.pack(...), " "))
  write(self,"\n")
  self:flush()
  return self
end

-- Forces to write in the pipe
function gnuplot_methods:flush()
  self.in_pipe:flush()
  return self
end

-- -- Plots (or multiplots) a given table with gnuplot parameters
-- function gnuplot_methods:plot(params, range)
--   -- remove previous temporal files
--   for _,tmpname in pairs(self.tmpnames) do
--     self:writeln(string.format("!rm -f %s", tmpname))
--   end
--   self.tmpnames = {}
--   local plot_str_tbl = {}
--   if range then
--     table.insert(plot_str_tbl, string.format("plot%s", range))
--   else
--     table.insert(plot_str_tbl, "plot")
--   end
--   local tmpnames = self.tmpnames
--   if not params[1] then params = { params } end
--   for i,current in ipairs(params) do
--     local data    = current.data
--     local func    = current.func
--     assert(data or func, "Field data or func is mandatory")
--     local using   = read("u", current.using or current.u)
--     local title   = read("title",current.title or current.t,"%q")
--     local notitle = read("notitle",current.notitle,"")
--     local with    = read("w",current.with or current.w)
--     local other   = current.other or ""
--     assert(type(other) == "string")
--     if type(data) == "matrix" or type(data) == "matrixDouble" then
--       local aux_tmpname = tmpnames[data]
--       if not aux_tmpname then
-- 	assert(data.toTabFilename,
-- 	       "The matrix object needs the method toTabFilename")
-- 	aux_tmpname = os.tmpname()
-- 	tmpnames[data] = aux_tmpname
-- 	data:toTabFilename(aux_tmpname)
--       end
--       data = aux_tmpname
--     end
--     if data then
--       if #data > 0 and string.sub(data,1,1) ~= "<" then
-- 	local f = assert(io.open(data), "Unable to open filename " .. data)
-- 	f:close()
--       end
--       data = string.format("%q", data)
--     end
--     table.insert(plot_str_tbl,
-- 		 string.format("%s %s %s %s %s",
-- 			       data or func, using, with, title, other))
--     if i ~= #params then table.insert(plot_str_tbl, ",") end
--   end
--   table.insert(plot_str_tbl, "\n")
--   -- print(table.concat(plot_str_tbl, " "))
--   write(self,table.concat(plot_str_tbl, " "))
--   self:flush()
--   return self
-- end

-- Plots (or multiplots) a given table with gnuplot parameters
function gnuplot_methods:plot(line, ...)
  assert(type(line) == "string",
         "New plot needs first the line string and varargs as arguments")
  assert(not line:find("^[%s]*plot[%s]*"),
         "New plot doesn't need 'plot' word in the given line string")
  local data = table.pack(...)
  -- remove previous temporal files
  for _,tmpname in pairs(self.tmpnames) do
    self:writeln(string.format("!rm -f %s", tmpname))
  end
  self.tmpnames = {}
  local tmpnames = self.tmpnames
  local dict = {}
  for i,m in ipairs(data) do
    local aux_tmpname = tmpnames[m]
    if not aux_tmpname then
      assert(m.toTabFilename,
	     "The matrix object needs the method toTabFilename")
      aux_tmpname = os.tmpname()
      tmpnames[m] = aux_tmpname
      m:toTabFilename(aux_tmpname)
    end
    dict["#"..i] = aux_tmpname
  end
  local line = line:gsub("(#%d*)",dict)
  -- print(line)
  self:writeln("plot " .. line)
  self:flush()
  return self
end

-- Closes the gnuplot pipe (interface)
function gnuplot_methods:close()
  -- remove previous temporal files
  for _,tmpname in pairs(self.tmpnames) do
    self:writeln(string.format("!rm -f %s", tmpname))
  end
  self.in_pipe:close()
  self.in_pipe  = nil
end

---------------
-- METATABLE --
---------------

------ METATABLE OF THE OBJECTS --------
local object_metatable = {}
object_metatable.__index = gnuplot_methods
object_metatable.__call  = gnuplot_methods.writeln
function object_metatable:__gc()
  self:close()
end

-----------------
-- CONSTRUCTOR --
-----------------

-- builds an instance for interfacing with gnuplot
function gnuplot.new()
  local f = io.popen("which gnuplot")
  local command = f:read("*l")
  f:close()
  assert(command, "Impossible to find gnuplot binary executable")
  local in_pipe= io.popen(command, "w")
  local obj = { in_pipe = in_pipe, tmpnames = {} }
  setmetatable(obj, object_metatable)
  return obj
end

-- gnuplot() is equivalent to gnuplot.new()
setmetatable(gnuplot, { __call = gnuplot.new })

do
  local singleton
  for k,v in pairs(gnuplot_methods) do
    assert(not gnuplot[k])
    gnuplot[k] = function(...)
      singleton = singleton or gnuplot.new()
      return singleton[k](singleton, ...)
    end
  end
end
-------------------
-- HELP FUNCTION --
-------------------

function gnuplot.help()
  print[[
Help of module 'gnuplot'.

This module is a wrap around gnuplot command, allowing
to make drawings of matrix objects (not lua tables),
filenames or gnuplot functions (expressions). The module
allow to declare as many gnuplot objects as you need,
none of them will share anything, so every object has
its own gnuplot window.


LOADING: load the module as usual

> gp = require "gnuplot"


CONSTRUCTOR: builds a gnuplot object instance

> gp = gnuplot()     -- the __call metamethod is defined
> gp = gnuplot.new() -- both are equivalent

HELP: shows this help message

> gnuplot.help()


METHOD __call: idem as below
METHOD WRITE LINE: writes a sentence line to gnuplot
  arguments:
    format: is a string with a printf format string
    ... : is a variable argument list

> gp(format, ...)
> gp:writeln(format, ...)


METHOD SET: executes the set command of gnuplot
  arguments:
    ... : a variable argument list, all of them strings

> gp:set("format x '%20f'")
> gp:set("xrange [-10,10]")


METHOD UNSET: executes the unset command of gnuplot
  arguments:
    ... : a variable argument list, all of them strings

> gp:unset("logscale y")


METHOD PLOT: allows to do generic plots, it uses placeholders #n to
use matrix objects as input.
  arguments:
    line string: a string with the gnuplot line (without the word 'plot')
    ...: vararg with matrix objects to be plotted

> gp:plot("'#1' u 3:4 w l lw 3", matrix_object) -- #1 is the placeholder
> gp:plot("3*x**2")
> gp:plot("'tmp.log' u 1:2 w l, '#1' u 1:2 w l, '' u 1:3 w l", m)


METHOD FLUSH: flushes all the pending data (normally it
              is not necessary)

> gp:flush()


METHOD CLOSE: closes the connection with gnuplot

> gp:close()


SINGLETON: it is possible to use gnuplot without calling the constructor, all
methods are available as static functions.

> gp.close()
> gp.rawplot(...)
> gp.set(...)
> ...

]]
end

return gnuplot
