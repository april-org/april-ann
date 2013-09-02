-- recibe un fichero indice y un directorio destino
index_list = arg[1]
dest_dir   = arg[2]

function load_matrix(MFCCfilename)
  local f = io.open(MFCCfilename) or
  error ("No se ha podido encontrar "..MFCCfilename)
  local aux = f:read("*a")
  local ad  = matrix.fromString(aux)
  f:close()
  return ad
end

gc_count = 1
for mat_filename in io.lines(index_list) do
  if gc_count > 100 then
    collectgarbage("collect")
    gc_count = 1
  else
    gc_count = gc_count+1
  end
  -- cargamos el dataset correspondiente a la frase actual
  -- local basename = string.remove_extension(string.basename(mat_filename))
  print ("# loading matrix", mat_filename)
  local mat = matrix.fromFilename(mat_filename)
  local basename = string.basename(mat_filename)
  local extension = string.get_extension(basename)
  basename  = string.remove_extension(basename)
  if (extension == 'gz') then
    basename  = string.remove_extension(basename)
  end
  local dest = dest_dir .. '/' .. basename .. '.htk'
  print("# writing file ",dest)
  htk_interface.write_matrix_as_mfc_file(mat,dest,"USER")
end
