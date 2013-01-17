23/04/2007
Aplicar los cambios del parche lua_dist2april.patch
---------------------------------------------------
cd lua-....
patch -p1 < ../lua_dist2april.patch

El p1 se salta el primer directorio de la ruta en la que se encuentran
los ficheros a parchear.

Cambios realizados a lua-5.1.2
------------------------------

Los mismos que a 5.1.1. He creado un parche con ambos cambios
para poder aplicarlos comodamente a futuras versiones (al menos de la
serie 5.1, supongo). No obstante, sigue siendo necesario especificar
las variables PLAT e INSTALL_TOP en el Makefile.


Cambios realizados a la version est√°ndar de lua-5.1.1
-----------------------------------------------------

En src/lib/liolib.c, funcion read_chars, cambio para evitar que un read termine interrumpido por seÒal y se entienda como un EOF

-  } while (n > 0 && nr == rlen);  /* until end of count or eof */
+  } while ((n > 0 && nr == rlen) || (ferror(f) && !feof(f) && errno==EINTR));  /* until end of count or eof */

 --

En el Makefile poner:
PLAT= posix
INSTALL_TOP= ../..

de esta manera instala lua en el directorio superior, quedando:

luapkg
|
|- lua
|  |
|  |- lua-5.1.1 <- hacer make y make install en este punto
|  |- lib
|  |- bin
|  |- include
|  \- man
|
.
.
.

--

Inicializacion de los paquetes en pmain en lua.c. Ya no existe llamada a
lua_userinit como la habia en la version anterior. PATCH:

--- /home/jgorbe/temp/lua-5.1.1/src/lua.c       2006-06-02 17:34:00.000000000 +0200
+++ lua/lua-5.1.1/src/lua.c     2007-03-15 18:44:51.000000000 +0100
@@ -344,6 +344,9 @@
   if (argv[0] && argv[0][0]) progname = argv[0];
   lua_gc(L, LUA_GCSTOP, 0);  /* stop collector during initialization */
   luaL_openlibs(L);  /* open libraries */
+#ifdef lua_userinit
+  lua_userinit(L);   /* APRIL: init packages */
+#endif
   lua_gc(L, LUA_GCRESTART, 0);
   s->status = handle_luainit(L);
   if (s->status != 0) return 0;



--------------------------------------------------------------------------------
------------ CAMBIOS DE LA 5.0.3 QUE NO SE HICIERON AL PASAR A 5.1 -------------
--------------------------------------------------------------------------------

En el fichero src/lib/liolib.c: AÒadida la funciÛn 'pclose'
para usar con el interprete de Lua. Las lÌneas aÒadidas est·n
m·rcadas con comentarios. Buscar 'Formiga' en el comentario.




En el fichero luac.c situado en src/luac hemos

a√±adido en la l√≠nea 93:

antes:

  else if (IS("-o"))			/* output file */
  {
   output=argv[++i];
   if (output==NULL || *output==0) usage("`-o' needs argument");
  }

ahora:

  else if (IS("-o"))			/* output file */
  {
   output=argv[++i];
   if (output==NULL || *output==0) usage("`-o' needs argument");
   if (IS("-")) output=NULL;
  }

y hemos cambiado la l√≠nea 182:

antes:

FILE* D=fopen(output,"wb");

ahora:

FILE* D= (output==NULL) ? stdout : fopen(output,"wb");

de esta manera se permite poner -o - para que saque el bytecode por salida est√°ndar.

Hemos cambiado tres l√≠neas en el fichero config:

descomentar para que funcione en el MacOS X:
POPEN= -DUSE_POPEN=1


descomentar:
MYCFLAGS= -O3 -fomit-frame-pointer # -fPIC

poner:
INSTALL_ROOT= ..

de esta manera instala lua en el directorio superior, quedando:

luapkg
|
|- lua
|  |
|  |- lua-5.0.2 <- hacer make y make install en este punto
|  |- lib
|  |- bin
|  |- include
|  \- man
|
.
.
.

