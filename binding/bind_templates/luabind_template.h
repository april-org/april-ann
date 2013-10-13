// $$FILENAME$$ 
// Generado automáticamente por luabind el $$os.date()$$ 
#ifndef LUA_BIND_$$FILENAME2$$_H
#define LUA_BIND_$$FILENAME2$$_H


extern "C" {
	#include <lua.h>
}

$$HEADER_H$$

//LUA for ClassName, class in pairs(CLASSES) do

#ifndef LUA_BIND_$$ClassName$$_$$FILENAME2$$_H
#define LUA_BIND_$$ClassName$$_$$FILENAME2$$_H

extern "C" {
  void bindluaopen_$$ClassName$$_$$FILENAME2$$(lua_State *L);
}

//LUA if CREATE_CLASS[ClassName] then
int lua_is$$ClassName$$(lua_State *L, int index);
$$ClassName$$ *lua_to$$ClassName$$(lua_State *L, int index);
void lua_push$$ClassName$$(lua_State *L, $$ClassName$$ *obj);
//LUA end

/*LUA 
-- las lineas de salida estandar que empiecen por "...register:" se utilizan para registrar ciertas funciones
oldprint("...register:bindluaopen_"..ClassName.."_"..FILENAME2)
*/

#endif
//LUA end


extern "C" {
  int lua_register_tables_$$FILENAME2$$(lua_State *L);
}

//LUA oldprint("...register:lua_register_tables_"..FILENAME2)



extern "C" {
  int lua_register_subclasses_$$FILENAME2$$(lua_State *L);
}

//LUA oldprint("...register:lua_register_subclasses_"..FILENAME2)

//LUA for name,code in pairs(STATIC_CONSTRUCTOR) do

extern "C" {
  int lua_execute_static_constructor_$$FILENAME2$$_$$name$$(lua_State *L);
}

//LUA oldprint("...register:lua_execute_static_constructor_"..FILENAME2.."_"..name)

//LUA end

#endif
