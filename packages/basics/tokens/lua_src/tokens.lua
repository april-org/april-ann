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
