// $$FILENAME$$ 
// Generado automáticamente por luabind el $$os.date()$$ 

/*LUA
  -- funcion para extraer las etiquetas //DOC_BEGIN .... //DOC_END
  function get_method_headers(code)
    local t = {}
    local list = {}
    -- buscamos cada grupo de etiquetas //DOC_BEGIN - //DOC_END
    for lines_headers in string.gmatch(code,
				       "[ \t]*%/%/DOC_BEGIN"..
					 "[^\n]*\n(.-)[ \t]*\n+"..
					 "%/%/DOC_END") do
      local i=1
      local funstr = ""
      local docstr = ""
      -- las tokenizamos por lineas
      for token in string.gmatch(lines_headers, '[^\n]+') do
	if i==1 then
	  -- la primera linea es la cabecera de la funcion
	  local type,func_hdr
	  ini,fin,type,func_hdr =
	  string.find(token,
	              "//%s*([^ \n]+)%s+([^\n]+)")
	  if type == nil then
	    ini,fin,func_hdr =
	      string.find(token, "//%s*([^\n]+)")
	    type = ""
	  end
	  funstr = type .. " " .. func_hdr
	  i = i + 1
	else
	  -- el resto es documentacion ;)
	  docstr = docstr .. "\n" .. token
	end
      end
      table.insert(list, docstr .. "\n" .. funstr)
    end
    return list
  end
 */

//LUA for ClassName, class in pairs(CLASSES) do

//LUA if CREATE_CLASS[ClassName] then

//LUA classname = LUANAME[ClassName] or ClassName

/*LUA
  -- extraemos el namespace: algo.otra.cosa...
  namespace  = ""
  last_token = nil
  for token in string.gmatch(classname, '[^%.]+') do
    if last_token then
*/
namespace $$last_token$$ {
/*LUA
    end
    last_token = token
  end
  doc_classname = last_token
 */

//LUA if not PARENT_CLASS[ClassName] then
class $$doc_classname$$ {
//LUA else
//LUA doc_parentclassname = string.gsub(LUANAME[PARENT_CLASS[ClassName]] or PARENT_CLASS[ClassName], "%.", "::")
//LUA parent_classname = 
  class $$doc_classname$$ : public $$doc_parentclassname$$ {
//LUA end
 public:
/*LUA
  method_headers = get_method_headers(class.constructor, classname);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
  $$func_header$$ {
  }
/*LUA
    end
  else
*/
  $$doc_classname$$(lua_State *L) {
  }
//LUA end


/*LUA
  method_headers = get_method_headers(class.destructor, classname);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
  $$func_header$$ {
  }
/*LUA
    end
  else
*/
  ~$$doc_classname$$(lua_State *L) {
  }
//LUA end

  void $$doc_classname$$_open_constructor(lua_State *L);
//LUA for MethodName,code in pairs(class.methods) do

/*LUA
  method_headers = get_method_headers(code);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
  $$func_header$$ {
  }
/*LUA
    end
  else
*/
  int $$MethodName$$(lua_State *L) {
  }
//LUA end

//LUA end
//LUA for ClassMethodName,code in pairs(class.class_methods) do

/*LUA
  method_headers = get_method_headers(code);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
  $$func_header$$ {
  }
/*LUA
    end
  else
*/
  static int $$ClassMethodName$$(lua_State *L) {
  }
//LUA end

//LUA end
};

/*LUA
  -- extraemos el namespace: algo.otra.cosa...
  namespace  = ""
  last_token = nil
  for token in string.gmatch(classname, '[^%.]+') do
    if last_token then
*/
}
/*LUA
    end
    last_token = token
  end
 */

//LUA end

//LUA end


//LUA for func_name,code in pairs(FUNCTIONS) do

/*LUA
  -- extraemos el namespace: algo.otra.cosa...
  namespace  = ""
  last_token = nil
  for token in string.gmatch(func_name, '[^%.]+') do
    if last_token then
*/
 namespace $$last_token$$ {
/*LUA
    end
    last_token = token
  end
  doc_func_name = last_token
 */

/*LUA
  method_headers = get_method_headers(code);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
$$func_header$$ {
}
/*LUA
    end
  else
*/
static int $$doc_func_name$$(lua_State *L) {
}
//LUA end

/*LUA
  -- extraemos el namespace: algo.otra.cosa...
  namespace  = ""
  last_token = nil
  for token in string.gmatch(func_name, '[^%.]+') do
    if last_token then
*/
}
/*LUA
    end
    last_token = token
  end
 */


//LUA end


//LUA for name,code in pairs(STATIC_CONSTRUCTOR) do

/*LUA
  method_headers = get_method_headers(code);
  if #method_headers > 0 then
    for i,func_header in ipairs(method_headers) do
*/
$$func_header$$ {
}
/*LUA
    end
  else
*/
int $$name$$_STATIC_CONSTRUCTOR(lua_State *L) {
}
//LUA end

//LUA end

#endif
