-- Fichero de test para el metodo arpa2lira ngram.stat
-- El fichero arpa no necesita estar ordenado
-- No devuelve nada, escribe el resultado en un fichero

function table.slice(t, ini, fin)
  local aux = {}
  for i=ini,fin do
    table.insert(aux, t[i])
  end
  return aux
end

--dofile("../../../basics/util/lua_src/trie.lua")
--dofile("../lua_src/arpa2lira.lua")

-- se limita a generar un fichero .lira
ngram.lira.arpa2lira{
  input_filename  = arg[1] or "dihana3gram.arpa",
  output_filename = arg[2] or "dihana3gram.lira",
  vocabulary      = lexClass.load(io.open("vocab"))
}

-- se limita a generar un fichero .lira
ngram.lira.arpa2lira{
  input_filename  = arg[1] or "mini.arpa",
  output_filename = arg[2] or "mini.lira",
  vocabulary      = lexClass.load(io.open("voc"))
}

