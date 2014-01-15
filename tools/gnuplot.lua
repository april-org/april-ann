local gnuplot = {} -- module gnuplot

--
local gnuplot_methods = {}

--
function gnuplot_methods:writef(format, ...)
  self.in_pipe:write(string.format(format,...))
  return self
end

--
function gnuplot_methods:set(...)
  self.in_pipe:write(table.concat(table.pack(...), " "))
  self.in_pipe:write("\n")
  return self
end

--
function gnuplot_methods:flush()
  self.in_pipe:flush()
end

--
function gnuplot_methods:plot(params)
  local tmpnames = self.tmpnames
  if not params[1] then params = { params } end
  self:writef("plot ")
  for i,current in ipairs(params) do
    local data    = current.data
    local using   = current.using or current.u or { 1 }
    local title   = current.title or current.t
    local notitle = current.notitle
    local with    = current.with or current.w or "points"
    local other   = current.other or ""
    assert(type(other) == "string")
    if type(using) ~= "table" then using = { using } end
    if type(data) == "matrix" then
      local aux_tmpname = os.tmpname()
      data:toTabFilename(aux_tmpname)
      data = aux_tmpname
      table.insert(tmpnames, aux_tmpname)
    end
    local title_str
    if title then
      title_str = string.format("title '%s'", title)
    else
      title_str = ""
    end
    if notitle then title_str = "notitle" end
    self:writef("'%s' u %s w %s %s %s",
		data, table.concat(using,":"), with, title_str, other)
    if i ~= #params then self:writef(",") end
  end
  self:writef("\n")
  self:flush()
end

--
function gnuplot_methods:close()
  self.in_pipe:close()
  for _,tmpname in ipairs(self.tmpnames) do
    os.remove(tmpnames[i])
  end
end

--
local object_metatable = {}

object_metatable.__index = gnuplot_methods

function object_metatable:__gc()
  self:close()
end

--
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

return gnuplot
