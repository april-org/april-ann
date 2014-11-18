-- GENERIC PRINT FUNCTION
matrix.__generic__ = matrix.__generic__ or {}

matrix.__generic__.__make_generic_print__ = function(name,getter)
  assert(name and getter)
  return function(self)
    local MAX_ROW_LEN,MAX_ROWS = 60,6
    local dims     = self:dim()
    local coords   = {}
    local out      = {}
    local row      = {}
    local row_len  = 0
    for i=1,#dims do coords[i]=1 end
    for i=1,self:size() do
      if #dims > 2 and coords[#dims] == 1 and coords[#dims-1] == 1 then
        table.insert(out,
                     string.format("\n# pos [%s]",
                                   table.concat(coords, ",")))
      end
      if not coords[#coords-1] or coords[#coords-1] < MAX_ROWS or coords[#coords-1] == dims[#coords-1] then
        if row_len < MAX_ROW_LEN or coords[#coords] == dims[#coords] then
          local str = getter(self:get(table.unpack(coords)))
          row_len = row_len + #str + 1
          if row_len < MAX_ROW_LEN  or coords[#coords] == dims[#coords] then
            table.insert(row, str)
          else
            table.insert(row, "...")
          end
        end
      elseif coords[#coords-1] == MAX_ROWS then
        row = { "..." }
      end
      local j=#dims+1
      repeat
        j=j-1
        coords[j] = coords[j] + 1
        if coords[j] > dims[j] then coords[j] = 1 end
      until j==1 or coords[j] ~= 1
      if coords[#coords] == 1 then
        if #row > 0 then table.insert(out, table.concat(row, " ")) row={} row_len = 0 end
      end
    end
    table.insert(out, string.format("# %s of size [%s] stride [%s] ref [%s]\n",
				    name,
				    table.concat(dims, ","),
                                    table.concat(self:stride(), ","),
				    self:get_reference_string()))
    return table.concat(out, "\n")
  end
end
