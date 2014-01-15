local gnuplot = {} -- module gnuplot

--------------- GNUPLOT OBJECT METHODS ---------------------
local gnuplot_methods = {}

-- Writes using format and a list of arguments
function gnuplot_methods:writef(format, ...)
  self.in_pipe:write(string.format(format,...))
  return self
end

-- Writes the given strings (separated by blanks)
function gnuplot_methods:write(...)
  self.in_pipe:write(table.concat(table.pack(...), " "))
  return self
end

-- Sets a parameter
function gnuplot_methods:set(...)
  self:write("set ")
  self:write(table.concat(table.pack(...), " "))
  self:write("\n")
  self:flush()
  return self
end

-- Forces to write in the pipe
function gnuplot_methods:flush()
  self.in_pipe:flush()
  return self
end

-- Plots (or multiplots) a given table with gnuplot parameters
function gnuplot_methods:plot(params, offset)
  local plot_str_tbl = { }
  if offset then
    table.insert(plot_str_tbl, string.format("plot%s", offset))
  else
    table.insert(plot_str_tbl, "plot")
  end
  local tmpnames = self.tmpnames
  if not params[1] then params = { params } end
  for i,current in ipairs(params) do
    local data    = assert(current.data, "Data field is mandatory")
    local using   = current.using or current.u or { 1 }
    local title   = current.title or current.t
    local notitle = current.notitle
    local with    = current.with or current.w
    local other   = current.other or ""
    assert(type(other) == "string")
    if type(using) ~= "table" then using = { using } end
    if type(data) == "matrix" or type(data) == "matrixDouble" then
      assert(data.toTabFilename,
	     "The matrix object needs the method toTabFilename")
      local aux_tmpname = os.tmpname()
      data:toTabFilename(aux_tmpname)
      data = aux_tmpname
      table.insert(tmpnames, aux_tmpname)
    end
    local title_str = ""
    if title then title_str = string.format("title '%s'", title) end
    if notitle then title_str = "notitle" end
    local with_str = ""
    if with then with_str = string.format("w %s", with) end
    table.insert(plot_str_tbl,
		 string.format("'%s' u %s %s %s %s",
			       data, table.concat(using,":"),
			       with_str, title_str, other))
    if i ~= #params then table.insert(plot_str_tbl, ",") end
  end
  table.insert(plot_str_tbl, "\n")
  self:write(table.concat(plot_str_tbl, " "))
  self:flush()
  return self
end

-- Closes the gnuplot pipe (interface)
function gnuplot_methods:close()
  self.in_pipe:close()
  for _,tmpname in ipairs(self.tmpnames) do
    os.remove(tmpname)
  end
  self.in_pipe  = nil
  self.tmpnames = {}
end

------ METATABLE OF THE OBJECTS --------
local object_metatable = {}
object_metatable.__index = gnuplot_methods
function object_metatable:__gc()
  self:close()
end

------------- CONSTRUCTOR --------------------

-- builds an instance for interfacing with gnuplot
function gnuplot.new()
  local f = io.popen("which gnuplot")
  local command = f:read("*l")
  f:close()
  assert(command, "Impossible to find gnuplot binary executable")
  local in_pipe,out_pipe = io.popen2(command)
  out_pipe:close()
  local obj = { in_pipe = in_pipe, tmpnames = {} }
  setmetatable(obj, object_metatable)
  return obj
end

-- gnuplot() is equivalent to gnuplot.new()
setmetatable(gnuplot, { __call = gnuplot.new })

return gnuplot
