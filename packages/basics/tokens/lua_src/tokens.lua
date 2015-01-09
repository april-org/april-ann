tokens = tokens or {}
tokens.table = tokens.table or {}
function tokens.table.row2colmajor(tbl)
  local t = {}
  for j=1,#tbl[1] do
    for i=1,#tbl do
      table.insert(t, tbl[i][j])
    end
  end
  return t
end

function tokens.table.colmajor2row(tbl, bunch_size)
  local out = {}
  for j=1,bunch_size do
    out[j] = {}
    for k=j,#tbl,bunch_size do
      table.insert(out[j],tbl[k])
    end
  end
  return out
end

---------------------------------------------------------------------------

class.extend(tokens.null, "to_lua_string",
             function(self)
               return "tokens.null()"
end)

class.extend(tokens.vector.bunch, "to_lua_string",
             function(self, format)
               local str = { "tokens.vector.bunch{" }
               for i,v in self:iterate() do
                 str[#str+1] = util.to_lua_string(v, format)
                 str[#str+1] = ","
               end
               str[#str+1] = "}"
               return table.concat(str)
end)
