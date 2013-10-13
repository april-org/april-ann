// $$FILENAME$$ 
// Generado automáticamente por luabind el $$os.date()$$ 
//
extern "C" {
	#include "lua.h"
	#include "lualib.h"
	#include "lauxlib.h"
}

#ifndef UNUSED_VARIABLE
#define UNUSED_VARIABLE(x) (void)(x)
#endif

#include "$$string.gsub(FILENAME,".*/([%w_]+).cc","%1.h")$$"
#include <cstring> // para strerror
#include <cstdlib> // para utilizar exit

#ifdef __GNUC__
#define UNUSED __attribute__((unused))
#else
#define UNUSED
#endif

#ifdef __DEBUG__
	#define DEBUG(a) printf("DEBUG %s\n",a);fflush(stdout)
	#define DEBUG_OBJ(a,o) printf("DEBUG(%p) %s\n",o,a);fflush(stdout) 
	#include <cstdio>
#else
	#define DEBUG(a)
	#define DEBUG_OBJ(a,o)
#endif

$$HEADER_C$$

#include "luabindutil.h"


/*LUA
  -- Esta funcion creara un nombre de tablas
  -- jerarquico, del tipo a.b.c, y dejara
  -- en el tope de la pila la ultima tabla
   function lua_setdottedname(dottedname)
      local cpp_str = {
                         "\nbool exists = false;\n",
			 "lua_pushglobaltable(L);\n"
	              }
      for subName in string.gmatch(dottedname, "([%w_]+)[%.]*") do
	 table.insert(cpp_str, string.format([[
  // stack: tablevieja
  lua_pushstring(L,"%s");
  lua_rawget(L,-2);
  if (lua_isnil(L,-1)) {
    exists = false;
    lua_pop(L,1);
    lua_newtable(L);
    lua_pushstring(L,"%s");
    lua_pushvalue(L,-2);
    // stack: tablanueva cadena tablanueva tablavieja ...
    lua_pushvalue(L,-1);
    // stack: tablanueva tablanueva cadena tablanueva tablavieja ...
    lua_setmetatable(L, -2);
    // stack: tablanueva cadena tablanueva tablavieja ...
    lua_rawset(L,-4);
  }
  else exists = true;
  //stack: tablanueva tablavieja
  lua_remove(L,-2);
  // stack: tablanueva

  ]], subName, subName))
      end
      return table.concat(cpp_str);
   end
*/


//LUA for ClassName, class in pairs(CLASSES) do

//LUA for MethodName in pairs(class.methods) do
int lua_call_$$ClassName$$_$$MethodName$$(lua_State *L);
//LUA end
//LUA for ClassMethodName in pairs(class.class_methods) do
int lua_call_class_$$ClassName$$_$$ClassMethodName$$(lua_State *L);
//LUA end
int lua_new_$$ClassName$$_$$FILENAME2$$(lua_State *L);
int lua_delete_$$ClassName$$_$$FILENAME2$$(lua_State *L);

$$ClassName$$ *lua_rawget$$ClassName$$_$$FILENAME2$$(lua_State *L, int index) {
	$$ClassName$$ **pre_obj = static_cast<$$ClassName$$**>(lua_touserdata(L,index));
	$$ClassName$$ *obj = 0;
	if (pre_obj) obj = (*pre_obj);
	return obj;
}

//LUA if CREATE_CLASS[ClassName] then

/* Implementacion :: */

$$ClassName$$ *lua_to$$ClassName$$(lua_State *L, int index){
	$$ClassName$$ *obj = lua_rawget$$ClassName$$_$$FILENAME2$$(L,index);
	
	DEBUG_OBJ("lua_to$$ClassName$$", obj);
	if (obj == NULL) return 0;
	if (!lua_getmetatable(L,index)) return 0;
	// metatabla
	lua_pushstring(L,"__index");
	lua_rawget(L,-2);
	// metatabla metainstance
	lua_pushstring(L,"is_$$(LUANAME[ClassName] or ClassName)$$");
	lua_gettable(L,-2);
	if ( !lua_isboolean(L,-1)) {
		obj = NULL;
	}
	lua_pop(L,3);
	
	return obj;
}

int lua_is$$ClassName$$(lua_State *L, int index){
  int result;
  $$ClassName$$ *obj = lua_rawget$$ClassName$$_$$FILENAME2$$(L,index);	
  DEBUG_OBJ("lua_is$$ClassName$$", obj);
  if (obj == NULL) return 0;
  if (!lua_getmetatable(L,index)) return 0;
  // metatabla
  lua_pushstring(L,"__index");
  lua_rawget(L,-2);
  // metatabla metainstance
  lua_pushstring(L,"is_$$(LUANAME[ClassName] or ClassName)$$");
  lua_gettable(L,-2);
  result = lua_isboolean(L,-1);
  lua_pop(L,3);
  return ( result != 0 );
}

void lua_push$$ClassName$$(lua_State *L, $$ClassName$$ *obj){
	$$ClassName$$ **ptr;
	
	DEBUG("lua_push$$ClassName$$ (begin)");
	ptr = static_cast<$$ClassName$$**>( 
	      lua_newuserdata(L,sizeof($$ClassName$$*)) ); // instance
	*ptr = obj;
	// pila =  ptr
	// le asociamos al userdata la metatabla que le corresponde
	lua_pushstring(L,"luabind_classes");
	// pila =  ptr, "luabind_clases"
	lua_rawget(L,LUA_REGISTRYINDEX);
	// pila =  ptr, luabind_clases
	lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$");
	// pila =  ptr, luabind_clases  ClassName
	lua_rawget(L,-2);
	// pila =  ptr, luabind_clases luabind_clases[ClassName]
	DEBUG_OBJ("lua_push$$ClassName$$",obj);
	lua_pushstring(L,"meta_instance");
	// pila =  ptr, luabind_clases luabind_clases[ClassName] "metainstance"
	lua_rawget(L,-2); // class_table
	// pila =  ptr, table:luabind_clases luabind_clases[ClassName] metainstance
	lua_setmetatable(L,-4); // obj_index
	// pila =  ptr, table:luabind_clases luabind_clases[ClassName]
	lua_pop(L,2);
	// pila = ptr
	
	//Hacemos un IncRef ya que estamos haciendo una nueva copia
	//del objeto
	IncRef(obj);
	DEBUG("lua_push$$ClassName$$ (end)");
}

//LUA end

// Macros para el constructor
#undef FUNCTION_NAME
#define FUNCTION_NAME "$$(LUANAME[ClassName] or ClassName)$$:$$(LUANAME[ClassName] or ClassName)$$"
#include "luabindmacros.h"

int lua_new_$$ClassName$$_$$FILENAME2$$(lua_State *L) {
	lua_remove(L,1);  // primer parametro es la metatabla __call(table,...)
	$$ClassName$$ UNUSED *obj = 0;
	DEBUG("lua_new_$$ClassName$$ (begin)");
        int luabind_num_returned_values = 0;
	$$class.constructor$$
	DEBUG("lua_new_$$ClassName$$ (end)");
        if (luabind_num_returned_values != 1) {
          LUABIND_ERROR("constructors must return 1 (and only 1) value");
        }
	return luabind_num_returned_values;
}

int lua_delete_$$ClassName$$_$$FILENAME2$$(lua_State *L){
  $$ClassName$$ *obj = lua_rawget$$ClassName$$_$$FILENAME2$$(L,1);
  if (obj != 0) {
    DEBUG_OBJ("lua_delete_$$ClassName$$ (begin)",obj);
    $$class.destructor$$
    DEBUG_OBJ("lua_delete_$$ClassName$$ (end)",obj);
    // Hacemos un DecRef para borrar la referencia a este objeto
    DecRef(obj);
  }
  // FIXME: This warning is due to the META_INSTANCE table, which is a metatable
  // of itself and has a __gc method, so at the end of APRIL execution, the __gc
  // is executed with a NULL pointer
  else {
    DEBUG_OBJ("lua_delete_$$ClassName$$ WARNING!! NULL pointer", obj);
  }
  return 0;
}


void bindluaopen_$$ClassName$$_$$FILENAME2$$(lua_State *L){
	int meta_instance;   // instance's metatable
	int meta_class;      // class's metatable
	int class_table;	 // tabla de la clase
	int instance_methods;// metodos

	DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (begin)");

	// Creamos la tabla para la clase
	$$lua_setdottedname(LUANAME[ClassName])$$
	// stack: last_table
	if (exists) {
	  DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (exists table)");
	  lua_pushstring(L,"luabind_classes");
	  lua_rawget(L,LUA_REGISTRYINDEX);
	  exists = false;
	  if (not lua_isnil(L,-1)) {
	    // stack: luabind_class class_table
	    lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$");
	    // stack: cadena luabind_class class_table
	    lua_rawget(L,-2);
	    // stack: value luabind_class class_table
	    if (not lua_isnil(L,-1)) {
	      // Ya existe esta clase en el registry
	      exists = true;
	    }
	    // stack: nil luabind_class class_table
	    lua_pop(L, 2);
	  }
	  else {
	    // stack: nil last_table
	    lua_pop(L, 1);
	  }
	}
	class_table = lua_gettop(L);
	
	if (not exists) {
	  DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (creating class)"); 
	  // creamos las tablas y guardamos los indices
	  lua_newtable(L);
	  meta_instance = lua_gettop(L);
	  lua_newtable(L);
	  meta_class = lua_gettop(L);
	  lua_newtable(L);
	  instance_methods = lua_gettop(L);
	  
	  // __tostring
	  lua_pushstring(L,"__tostring");
	  lua_pushcfunction(L,lua_print_name_instance);
	  lua_settable(L,meta_instance);
	  lua_pushstring(L,"__tostring");
	  lua_pushcfunction(L,lua_print_name_class);
	  lua_settable(L,meta_class);
	  
	  // Class metatable
	  lua_pushstring(L,"__call");
	  lua_pushvalue(L,class_table);
	  lua_pushcclosure(L, lua_new_$$ClassName$$_$$FILENAME2$$,1);
	  lua_settable(L,meta_class);
	  
	  // Class table
	  lua_pushstring(L,"meta_instance");
	  lua_pushvalue(L,meta_instance);
	  lua_settable(L,class_table);
	  
	  // Meta_instance
	  lua_pushstring(L,"__gc");
	  lua_pushcfunction(L,lua_delete_$$ClassName$$_$$FILENAME2$$);
	  lua_settable(L,meta_instance);
	  lua_pushstring(L,"__eq");
	  lua_pushcfunction(L,equaluserdata);
	  lua_settable(L,meta_instance);
	  lua_pushstring(L,"__index");
	  lua_pushvalue(L,instance_methods);
	  lua_settable(L,meta_instance);
	  lua_pushstring(L,"is_$$(LUANAME[ClassName] or ClassName)$$");  //table's (class) id...
	  lua_pushboolean(L,1);
	  lua_settable(L,instance_methods);
	  lua_pushstring(L,"id");  //table's (class) id...
	  lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$");
	  lua_settable(L,meta_instance);

	  // todo: revisar si dejamos esto:
	  lua_pushstring(L,"id");  //table's (class) id...
	  lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$ class");
	  lua_settable(L,meta_class);


	  // class metatable
	  lua_pushvalue(L,meta_class);
	  lua_setmetatable(L,class_table);
	  
	  
	  // luabind_classes at RegistryIndex
	  lua_pushstring(L,"luabind_classes");
	  lua_rawget(L,LUA_REGISTRYINDEX);
	  if (lua_isnil(L,-1)){
	    lua_pop(L,1); 
	    lua_newtable(L);
	    lua_pushstring(L,"luabind_classes");
	    lua_pushvalue(L,-2);
	    lua_rawset(L,LUA_REGISTRYINDEX);
	  }
	  
	  lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$");
	  lua_pushvalue(L,class_table);
	  lua_rawset(L,-3);
	  /* Esto no hace falta
	     lua_pushstring(L,"$$(LUANAME[ClassName] or ClassName)$$");
	     lua_pushvalue(L,class_table);
	     lua_rawset(L,LUA_RIDX_GLOBALS);
	  */
	  
	  lua_pop(L,1);  // luabind_classes
	} else {
	  DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (acess created class)"); 
	  // Ya existe la clase, buscamos las tablas
	  // stack: class_table
	  lua_pushstring(L,"meta_instance");
	  lua_gettable(L, class_table);
	  // stack: meta_instance class_table
	  meta_instance = lua_gettop(L);
	  lua_pushstring(L,"__index");
	  lua_gettable(L, meta_instance);
	  // stack: instance_methods meta_instance class_table
	  instance_methods = lua_gettop(L);
	}
	
	DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (methods)"); 
	// Instance_methods
	//LUA for MethodName in pairs(class.methods) do
	lua_pushstring(L,"$$MethodName$$");
	lua_pushcfunction(L,lua_call_$$ClassName$$_$$MethodName$$);
	lua_settable(L,instance_methods);
	//LUA end
	
	// class_methods
	//LUA for ClassMethodName in pairs(class.class_methods) do
	lua_pushstring(L,"$$ClassMethodName$$");
	lua_pushcfunction(L,lua_call_class_$$ClassName$$_$$ClassMethodName$$);
	lua_settable(L,class_table);
	//LUA end
	
	if (exists) {
	  // Eliminamos las tablas que hemos puesto en la pila
	  lua_pop(L, 3);
	}
	else {
	  lua_pop(L,4); // created tables
	}

	//LUA if class.class_open ~= ""  then 
	// Codigo a ejecutar cuando se hace open de la clase
	DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (open)");
	{
		$$class.class_open$$
	} 

	//LUA end
	DEBUG("bindluaopen_$$ClassName$$_$$FILENAME2$$ (end)");
}

//LUA for MethodName,code in pairs(class.methods) do
#undef FUNCTION_NAME
#define FUNCTION_NAME "$$(LUANAME[ClassName] or ClassName)$$:$$MethodName$$"
#include "luabindmacros.h"

int lua_call_$$ClassName$$_$$MethodName$$(lua_State *L){
  //Comprobamos que el primer elemento sea el userdata que esperamos
  if (!lua_is$$ClassName$$(L,1)) {
      lua_pushstring(L, "First argument of $$MethodName$$ must be of type "
                        "$$ClassName$$. Did you use '.' instead of ':'?");
      lua_error(L);
  }
  $$ClassName$$ *obj = lua_rawget$$ClassName$$_$$FILENAME2$$(L,1);
  int luabind_num_returned_values = 0;
  DEBUG_OBJ("lua_call_$$ClassName$$_$$MethodName$$ (begin)", obj);
  lua_remove(L,1); // methods do not need the instance pointer	
  // CODE:
  {
    $$code$$
      }
  DEBUG_OBJ("lua_call_$$ClassName$$_$$MethodName$$ (end)", obj);
  return luabind_num_returned_values;
}
//LUA end

//LUA for ClassMethodName,code in pairs(class.class_methods) do
#undef FUNCTION_NAME
#define FUNCTION_NAME "$$(LUANAME[ClassName] or ClassName)$$:$$ClassMethodName$$"
#include "luabindmacros.h"

int lua_call_class_$$ClassName$$_$$ClassMethodName$$(lua_State *L){
	
	DEBUG("lua_call_class_$$ClassName$$_$$ClassMethodName$$");
        int luabind_num_returned_values = 0;
	// CODE:
	{
	  $$code$$
	}
	return luabind_num_returned_values;
}
//LUA end

//LUA end


//LUA for func_name,code in pairs(FUNCTIONS) do
#undef FUNCTION_NAME
#define FUNCTION_NAME "$$func_name$$"
#include "luabindmacros.h"

static int lua_call_$$string.gsub(func_name,"%p","_")$$(lua_State *L){
  lua_remove(L,1);  // primer parametro es la metatabla __call(table,...)
  DEBUG("lua_call_$$string.gsub(func_name,"%p","_")$$ (begin)");
  int luabind_num_returned_values = 0;
  $$code$$
  DEBUG("lua_call_$$string.gsub(func_name,"%p","_")$$ (end)");
  return luabind_num_returned_values;
}
//LUA end

int lua_register_tables_$$FILENAME2$$(lua_State *L){
  UNUSED_VARIABLE(L);
  DEBUG("lua_register_tables_$$FILENAME2$$ (begin)");
  //LUA for table_name in pairs(TABLES) do
  {
    $$lua_setdottedname(table_name)$$
  }
  lua_pop(L,1);
  //LUA end
  //LUA for func_name in pairs(FUNCTIONS) do
  {
    $$lua_setdottedname(func_name)$$
  }
  // stack: table
  lua_pushvalue(L,-1);
  // stack: table table
  lua_setmetatable(L,-2);
  // stack: table
  lua_pushstring(L, "__call");
  // stack: cadena table
  lua_pushcfunction(L, lua_call_$$string.gsub(func_name,"%p","_")$$);
  // stack: funcion cadena table
  lua_rawset(L, -3);
  // stack: table
  lua_pop(L,1);
  //LUA end
  //LUA for varsTblName,vars in pairs(ENUM_CONSTANT) do
  {
    $$lua_setdottedname(varsTblName)$$
  }
  //LUA for varName,varValue in pairs(vars) do
  // stack: table
  lua_pushstring(L, "$$varName$$");
  // stack: string table
  lua_pushnumber(L, $$varValue$$);
  // stack: value string table
  lua_rawset(L, -3);
  // stack: table
  lua_pushnumber(L, $$varValue$$);
  // stack: value table
  lua_rawget(L, -2);
  if (lua_isnil(L, -1)) {
    // stack: nil table
    lua_pop(L,1);
    // stack: table
    lua_pushnumber(L, $$varValue$$);
    // stack: value table
    lua_pushstring(L, "$$varName$$");
    // stack: string value table
    lua_rawset(L, -3);
    // stack: table
  }
  else {
    fprintf(stderr, "ENUM_CONSTANT value is repeated: $$varsTblName$$.$$varName$$\n");
    exit(10);
  }
  //LUA end
  lua_pop(L,1);
  //LUA end


  //LUA for varsTblName,vars in pairs(STRING_CONSTANT) do
  {
    $$lua_setdottedname(varsTblName)$$
  }
  //LUA for varName,varValue in pairs(vars) do
  // stack: table
  lua_pushstring(L, "$$varName$$");
  // stack: string table
  lua_pushstring(L, $$varValue$$);
  // stack: value string table
  lua_rawset(L, -3);
  // stack: table
  lua_pushstring(L, $$varValue$$);
  // stack: value table
  lua_rawget(L, -2);
  if (lua_isnil(L, -1)) {
    // stack: nil table
    lua_pop(L,1);
    // stack: table
    lua_pushstring(L, $$varValue$$);
    // stack: value table
    lua_pushstring(L, "$$varName$$");
    // stack: string value table
    lua_rawset(L, -3);
    // stack: table
  }
  else {
    fprintf(stderr, "STRING_CONSTANT value is repeated: $$varsTblName$$.$$varName$$\n");
    exit(10);
  }
  //LUA end
  lua_pop(L,1);
  //LUA end

  DEBUG("lua_register_tables_$$FILENAME2$$ (end)");
  return 0;
}

int lua_register_subclasses_$$FILENAME2$$(lua_State *L){
  UNUSED_VARIABLE(L);
  DEBUG("lua_register_subclasses_$$FILENAME2$$ (begin)");
  //LUA for childclass, parentclass in pairs(PARENT_CLASS) do
  
  // buscamos la metatabla de la clase child
  lua_pushstring(L,"luabind_classes");
  // pila =  ... "luabind_clases"
  lua_rawget(L,LUA_REGISTRYINDEX);
  // pila =  ... luabind_clases  
  lua_pushstring(L,"$$(LUANAME[childclass] or childclass)$$");
  // pila =  ... luabind_clases childclass
  lua_rawget(L,-2);
  // pila =  ... luabind_clases luabind_clases[childclass]
  lua_pushstring(L,"meta_instance");
  // pila =  ... luabind_clases luabind_clases[childclass] "metainstance"
  lua_rawget(L,-2); // class_table
  // pila =  ... luabind_clases luabind_clases[childclass] child_metainstance
  lua_pushstring(L,"__index");
  // pila =  ... luabind_clases luabind_clases[childclass] child_metainstance "__index"
  lua_rawget(L,-2);
  // pila =  ... luabind_clases luabind_clases[childclass] child_metainstance child_metainstance[__index]
  lua_remove(L,-2);
  // pila =  ... luabind_clases luabind_clases[childclass] child_metainstance[__index]

  lua_pushstring(L,"$$(LUANAME[parentclass] or parentclass)$$");
  // pila =  ... luabind_clases luabind_clases[childclass] child_metainstance[__index] parentclass_string
  lua_rawget(L,-4); // -4 deberia ser luabind_clases
  // pila = ... luabind_clases luabind_clases[childclass]
  // child_metainstance[__index] luabindclasses[parentclass]
  lua_pushstring(L,"meta_instance");
  // pila = ... luabind_clases luabind_clases[childclass]
  // child_metainstance[__index] luabindclasses[parentclass] "meta_instance"
  lua_rawget(L,-2); // class_table
  // pila = ... luabind_clases luabind_clases[childclass]
  // child_meta_instance[__index] luabindclasses[parentclass] parent_meta_instance

  lua_setmetatable(L,-3);
  // pila = ... luabind_clases luabind_clases[childclass]
  // child_meta_instance[__index] luabindclasses[parentclass]
  lua_pop(L,4);
  // pila =  ...

  //LUA end
  DEBUG("lua_register_subclasses_$$FILENAME2$$ (end)");
  return 0;
}

//LUA for name,code in pairs(STATIC_CONSTRUCTOR) do

int lua_execute_static_constructor_$$FILENAME2$$_$$name$$(lua_State *L) {
  UNUSED_VARIABLE(L);
  DEBUG("lua_execute_static_constructor_$$FILENAME2$$_$$name$$ (begin)");
  $$code$$
  DEBUG("lua_execute_static_constructor_$$FILENAME2$$_$$name$$ (end)");
  return 0;
}

//LUA end
