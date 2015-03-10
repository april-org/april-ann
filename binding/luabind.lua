-- Variable GLOBAL para conrolar que se generen numeros de linea
-- utiles para GCC

-- strip tipico 
--    simbolos = simbolos que actuan de separadores
function strip(texto,simbolos)
   local SEPARADOR_EXT="["..simbolos.."]+"
   local res = {}
   local total = string.len(texto)
   local last = 1
   local pos ,fin
   pos,fin = string.find(texto,SEPARADOR_EXT, last)
   while pos do
      table.insert(res, string.sub(texto,last,pos-1))
      last = fin + 1
      pos,fin = string.find(texto,SEPARADOR_EXT, last)
   end
   if last < total then
      table.insert(res,string.sub(texto, last, total))
   end

   return res
end

-- Funcion para procesar texto, devuelve un texto que puede ser
-- ejecutado para procesar el fichero.
function expand_text(original)
   local inicio_code = nil
   local fin_marca = nil
   local tipo = nil
   local last = 1
   local total = string.len(original)
   local resultado = ""
   local fin_marca_code 
   
   -- Dentro del print de codigo normal se buscan cadenas encerradas entre '$$'
   -- que serán ejecutadas en lua ( para concatenar como cadena de texto )
   local function parse(text) 
      if text=="" then return "" end
      local res = " print([["..text.."]]) "
      res = string.gsub(res,"%$%$([^$]+)%$%$","]]..%1..[[")
      return res
   end

   -- Buscamos comentarios del tipo /*LUA o //LUA y al hacer
   -- /([*/])LUA el tercer argumento devuelto sera "*" o "/"
   -- que determina el tipo de comentario

   inicio_code, fin_marca, tipo= string.find(original, "/([*/])LUA",last)
   while inicio_code ~= nil do
      if tipo == '/' then 
	 fin_code, fin_marca_code = string.find(original, "\n",fin_marca)
      else
	 fin_code, fin_marca_code = string.find(original,"%*/",fin_marca)
      end
      --  Escribimos --
      resultado = resultado..parse(string.sub(original,last,inicio_code-1))
      resultado = resultado..string.sub(original,fin_marca+1,fin_code-1)
      
      last = fin_marca_code+1
      inicio_code, fin_marca, tipo= string.find(original, "/([*/])LUA",last)
   end

   if (last < total) then 
      resultado = resultado..parse(string.sub(original,last,total))
   end

   return resultado
end

-- para cargar tal cual un fichero en una cadena
function load_raw(filename)
   local f = io.open(filename)
   local c = f:read("*a")
   f:close()
   return c
end

-- load_data carga información del fichero filename y fija ciertas variables
-- globales: 
--   CLASSES       : tabla indexada por nombre de Clase y que contiene:
--        [nombre_clase] = {
--                 methods = {}          ( [nombre_metodo] = string (codigo) )
--                 class_methods = {}    ( [nombre_metodo] = string (codigo) )
--                 class_open   = string (codigo)
--                 constructor  = string ( codigo )
--                 destructor   = string ( codigo )
--               }
--   HEADER_H        : Cabecera del posible .h
--   HEADER_C        : Cabecera del posible .cc
--   FOOTER_H        : Pie del posible .h
--   FOOTER_C        : Pie del posible .cc

FUNCTIONS = {}
TABLES = {}
CREATE_CLASS  = {}
CLASSES = {}
HEADER_H = ""
FOOTER_H = ""
HEADER_C = ""
FOOTER_C = ""
LUANAME = {}
PARENT_CLASS = {}
STATIC_CONSTRUCTOR = {}
ENUM_CONSTANT = {}
STRING_CONSTANT = {}
function load_data(filename)
   local f,_error = io.open(filename)
   if (f == nil) then
     print("error opening file ".. filename .." : ".. (_error or "nil"))
      os.exit()
   end
   local line, ini, fin, name
   local num_line = 0
   local buffer = {}
   local last_table = nil
   local last_key = ""
   local last_className = ""

   
   -- Guardar en la ultima tabla,key el buffer actual
   local function store_buffer() 
      if last_table and last_key then
	 if lines_information == nil then
	   table.insert(buffer,'/*___LUA_END___ ('..last_className..'::'..last_key..') '..
			filename..":"..(num_line-1)..' *****/')
	 end
	 last_table[last_key] = table.concat(buffer,'\n')
	 last_table = nil
	 last_key = ""
	 buffer = {}
      end
   end

   -- inicializar el buffer de codigo fuente
   local function store_header()
      if lines_information == nil then
	table.insert(buffer,'\n/*__LUA_BEGIN__ ('..last_className..'::'..last_key..') '..
		     filename..":"..(num_line+1)..' *****/\n#line '..
		       (num_line+1)..' "'..filename..'"')
      end
   end

   -- Comprueba si una clase existe y si no la inicializa
   local function check_class(ClassName)
     if not CLASSES[ClassName] then
	if not LUANAME[ClassName] then LUANAME[ClassName] = ClassName end
	 CLASSES[ClassName] = {
	    methods={},
            lua_to_hook = "",
	    constructor = "",
	    destructor  = "",
	    class_open = "",
	    class_methods = {}
	 }
      end
   end
   
   -- Funciones para cada una de las marcas
   local dolabel = {
      STATIC_CONSTRUCTOR = function(name)
	assert(name, "STATIC_CONSTRUCTOR needs a name")
			      last_className = "static_constructor_for_"..name
			      last_table, last_key = STATIC_CONSTRUCTOR, name
			      store_header()
			      print("...Static Constructor::"..name)
			   end,
      CREATE_TABLE = function(TableName)
	assert(TableName, "CREATE_TABLE needs a TableName")
      		        TABLES[TableName] = true
			last_table = TABLES
			last_key = TableName
			print ("...Create Table "..TableName)
		     end,
      FUNCTION = function(TableName)
	assert(TableName, "FUNCTION needs a TableName")
		    last_table = FUNCTIONS
		    last_className = TableName
		    last_key = TableName
		    store_header ()
		    print ("...Function "..TableName)
		 end,
      METHOD = function(ClassName, Method)
	assert(ClassName and Method, "METHOD needs a ClassName and Method")
		  check_class(ClassName)
		  last_className = ClassName
		  last_table, last_key = CLASSES[ClassName].methods, Method
		  store_header()
		  print("..."..ClassName.."::"..Method)
	       end,
      CLASS_METHOD = function(ClassName, Method) 
	assert(ClassName and Method, "CLASS_METHOD needs a ClassName and Method")
			check_class(ClassName)
			last_className = ClassName
			last_table, last_key = CLASSES[ClassName].class_methods, Method
			store_header()
			print("..."..ClassName.."::"..Method.."(class_method)")
		     end,
      CLASS_OPEN = function(ClassName)
	assert(ClassName, "CLASS_OPEN needs a ClassName")
		      check_class(ClassName)
		      last_className = ClassName
		      last_table, last_key = CLASSES[ClassName], "class_open"
		      store_header()
		      print("..."..ClassName.." (open)")
		   end,
      CONSTRUCTOR = function(ClassName) 
	assert(ClassName, "CONSTRUCTOR needs a ClassName")
		       check_class(ClassName)
		       last_className = ClassName
		       last_table, last_key = CLASSES[ClassName], "constructor"
		       store_header()
		       print("..."..ClassName.."::Constructor")
		    end,
      DESTRUCTOR = function(ClassName) 
	assert(ClassName, "DESTRUCTOR needs a ClassName")
		      check_class(ClassName)
		      last_className = ClassName
		      last_table, last_key = CLASSES[ClassName], "destructor"
		      store_header()
		      print("..."..ClassName.."::Destructor")
		   end,	
      END = function() 
	       store_buffer()
	    end,
      HEADER_C = function() 
		    last_className ="Cabecera"
		    last_table, last_key = _G, "HEADER_C"
		    store_header()
		    print("...Header cc")
		 end,
      HEADER_H = function() 
		    last_className ="Cabecera"
		    last_table, last_key = _G, "HEADER_H"
		    store_header()
		    print("...Header h")
		 end,
      FOOTER_H = function() 
		    last_className ="Pie"
		    last_table, last_key = _G, "FOOTER_H"
		    store_header()
		    print("...Footer h")
		 end,
      FOOTER_C = function() 
		    last_className ="Pie"
		    last_table, last_key = _G, "FOOTER_C"
		    store_header()
		    print("...Footer c")
		 end,
      CLASS_LUA_TO_HOOK = function(ClassName)
                  assert(ClassName, "CLASS_LUA_TO_HOOK needs a ClassName")
                  check_class(ClassName)
		  last_className = ClassName
		  last_table, last_key = CLASSES[ClassName], "lua_to_hook"
		  store_header()
		  print("...lua_to_hook "..ClassName)
	       end,
      LUACLASSNAME = function (cName, luaName)
	assert(cName and luaName, "LUACLASSNAME needs a cName and luaName")
			LUANAME[cName] = luaName
		     end,
      CPP_CLASS = function (cName)
	assert(cName, "CPP_CLASS needs a cName")
      		     CREATE_CLASS[cName] = true
		  end,

      SUBCLASS_OF = function(childclass,parentclass)
	assert(childclass and parentclass, "SUBCLASS_OF needs a childclass and parentclass")
		       if PARENT_CLASS[childclass] then
			  error ("Redefining a SUBCLASS_OF label in "..
				 "binding trying '".. childclass .."' is child of '"..
				 parentclass .."'!!!")
		       end
		      PARENT_CLASS[childclass] = parentclass
		      -- CLASSES[parentclass] metatabla de CLASSES[childclass]
		    end,
      
      -- Esta macro permite exportar a Lua valores de ENUMS o MACROS
      -- de C/C++. OJO!!! al ser ENUMS, los valores siempre hacen
      -- referencia a CONSTANTES NUMERICAS (nunca numeros), y no se
      -- permite que dos o mas compartan el mismo valor.
      ENUM_CONSTANT = function(varName, varValue)
	assert(varValue and varValue, "ENUM_CONSTANT needs a varName and varValue")
			local tbl=string.gsub(varName, "^(.*)%.[^%.]*$", "%1")
			local name=string.gsub(varName, "^.*%.([^%.]*)$", "%1")
			ENUM_CONSTANT[tbl] = ENUM_CONSTANT[tbl] or {}
			if ENUM_CONSTANT[tbl][name] then
			  io.stderr:write("ENUM_CONSTANT " .. varName .. " is repeated!!!\n")
			  os.exit(10)
			end
			ENUM_CONSTANT[tbl][name] = varValue
		      end,

      -- Esta macro permite exportar a Lua valores de STRINGS o MACROS
      -- de C/C++. OJO!!! al ser CONSTANTES, los valores siempre hacen
      -- referencia a CONST CHAR *, y no se
      -- permite que dos o mas compartan el mismo valor.
      STRING_CONSTANT = function(varName, varValue)
	assert(varValue and varValue, "STRING_CONSTANT needs a varName and varValue")
			local tbl=string.gsub(varName, "^(.*)%.[^%.]*$", "%1")
			local name=string.gsub(varName, "^.*%.([^%.]*)$", "%1")
			STRING_CONSTANT[tbl] = STRING_CONSTANT[tbl] or {}
			if STRING_CONSTANT[tbl][name] then
			  io.stderr:write("STRING_CONSTANT " .. varName .. " is repeated!!!\n")
			  os.exit(10)
			end
			STRING_CONSTANT[tbl][name] = varValue
		      end,

    }
   
   for line in f:lines() do
      num_line = num_line + 1
      ini,fin,name,args = string.find(line,"[ \t]*//BIND_([^ ]*) *(.*)")
      if ini ~= nil then
	 if dolabel[name] then 
	    args = strip(args," ")
	    dolabel[name](unpack(args))
	 else
	    io.stderr:write("Error: Etiqueta "..name.." Desconocida\n")
	    os.exit(1)
	 end -- if dolabel[name]
      else 
	table.insert(buffer,line)
      end -- if ini
      
   end -- for

   store_buffer() 
end

-- MAIN
if #arg < 3 then
   io.stderr:write(
		   "USO: "..arg[-1].." "..arg[0].." file_output plantilla fichero_to_bind1 [fichero_to_bind2 ...]\n")
   os.exit(1)
end

f = io.open(arg[2])
plantilla = expand_text( f:read("*a") )
for i=3,#arg do
   load_data( arg[i] )
end

FILENAME = arg[1]
_,_,FILENAME2 = string.find(FILENAME,"[.]*/([%w_]+)%.[^/]*$")
FILENAME2 = string.gsub(FILENAME2,"%p", "_")

if FILENAME==arg[2] or FILENAME==arg[3] then
   io.stderr:write(
		   "El fichero a generar coincide con la plantilla o el fuente ["..
		      FILENAME.."]\n\tPlantilla="..arg[2].."\n\tFuente="..arg[3].."\n")
   os.exit(1)
end
-- redirigimos la salida
new_output = io.open(FILENAME,"w+")
if new_output == nil then
  print("Error: file "..FILENAME.." cannot be created")
  os.exit(256)
end
oldprint = print
print = function(...)
	   new_output:write(...,"\n")
	end 

-- ejecutamos
--f=io.open("wop.tmp","w")
--f:write(plantilla) -- para debug
--f:close()
assert(loadstring(plantilla))()
