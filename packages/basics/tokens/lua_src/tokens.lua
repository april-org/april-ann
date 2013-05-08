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
