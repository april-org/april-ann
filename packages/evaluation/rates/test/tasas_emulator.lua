--[[

TODO: usar cmdOpt y tasas para emular el tasas original:

Uso: ./tasas fich [-f "c"] [-s #|-s "c"] [-p #] [TASA] [-c mat|-C mat] [-w dicc] [-v]
donde: 
fich es el fichero de datos
  -f "c"   : hace que c separe frases en una linea
  -s #     : hace que cada # caracteres se consideren un simbolo
  -s "cad" : caracts. en "cad" separan simbolos
  -p #     : fija el parametro p a #
  TASA     : proporciona una tasa concreta
             [-pra|-pre|-pa|-ip|-ie|-psb|-iep|-iap] 
  -C mat   : guarda en fichero mat la matriz de confusion
  -c mat   : guarda en mat elementos no nulos de matriz de confusion
  -w dicc  : toma del fichero dicc el orden de los simbolos 
  -v       : muestra el numero de ops. de cada tipo y valor de p usado
Si el nombre de un fichero es "-" se toma la entrada/salida estandar
POR DEFECTO: ./tasas -f "*" -s 1 -pra

--]]

cmdOptTest = cmdOpt{
  program_name = "tasas",
  argument_description = "emulador del programa tasas",
  main_description = "",
  author = "",
  copyright = "",
  see_also = "",
  -- options can be added in the constructor:
  { index_name="words_sep",
    description = "hace que c separe frases en una linea",
    long = "words_sep",
    short = "f",
    argument = "yes",
    argument_name = '"c"',
  },
  { index_name="words_width",
    description = "hace que cada # caracteres se consideren un simbolo",
    long = "words_width",
    short = "s",
    argument = "yes",
    argument_name = "#",
  },
  { index_name="p",
    description = "fija el parametro p a #",
    short = "p",
    argument = "yes",
  },
  { index_name="pra",
    description = "list directory entries instead of contents, and do not dereference symbolic links",
    short = "d",
    long = "directory",
    argument = "no",
  },
}
-- options can also be declared outside constructor:
cmdOptTest:add_option{
  description = "shows this help message",
  short = "h",
  long = "help",
  argument = "no",
  action = function (argument) 
	     print(cmdOptTest:generate_help()) 
	     os.exit(1)
	   end
}
cmdOptTest:add_option{
  index_name = "ignore-backups",
  description = "do not list implied entries ending with ~",
  short = "B",
  long = "ignore-backups",
  argument = "no",
  action = function(value) print("this is a test "..tostring(value)) end,
}
cmdOptTest:add_option{
  index_name = "ignore",
  description = "do not list implied entries matching shell PATTERN",
  short = "I",
  long = "ignore",
  argument = "yes",
  argument_name = "PATTERN",
}

----------------------------------------------------------------------
---------------------------- MAIN PROGRAM ----------------------------
----------------------------------------------------------------------
result = cmdOptTest:parse_args()
if type(result) == 'string' then error(result) end

print("\nThis is a test, let's see the contents of result table:")
for i,j in pairs(result) do
  print(i,j)
end

