2014/11/17: Updated with Lua 5.3.0

- Added lua-5.3.0.patch andd lstrip-5.3.0.patch files.

$ diff -rupN ~/programas/lua-5.3.0-beta/  lua/lua-5.3.0/  > lua/lua-5.3.0.patch
$ diff -rupN ~/programas/lstrip/  lua/lstrip/  > lua/lstrip-5.3.0.patch

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

2013/10/13: COPYRIGHT, AUTHORS, and VERSION stuff is out of Lua source tree, it
is managed by formiga.

2013/09/17: Modified lua.h to incorporate the commit number

diff --git a/lua/lua-5.2.2/src/lua.h b/lua/lua-5.2.2/src/lua.h
index c675516..027bc17 100644
--- a/lua/lua-5.2.2/src/lua.h
+++ b/lua/lua-5.2.2/src/lua.h
@@ -26,10 +26,15 @@
 #define LUA_COPYRIGHT	LUA_RELEASE "  Copyright (C) 1994-2013 Lua.org, PUC-Rio"
 #define LUA_AUTHORS	"R. Ierusalimschy, L. H. de Figueiredo, W. Celes"
 
+#ifndef APRILANN_COMMIT
+#define APRILANN_COMMIT UNKNOWN
+#endif
+#define STRINGFY(X) #X
+#define TOSTRING(X) STRINGFY(X)
 #define APRILANN_VERSION_MAJOR "0"
 #define APRILANN_VERSION_MINOR "2"
 #define APRILANN_VERSION_RELEASE "1"
-#define APRILANN_RELEASE   "April-ANN v" APRILANN_VERSION_MAJOR "." APRILANN_VERSION_MINOR "." APRILANN_VERSION_RELEASE "-beta"
+#define APRILANN_RELEASE   "April-ANN v" APRILANN_VERSION_MAJOR "." APRILANN_VERSION_MINOR "." APRILANN_VERSION_RELEASE "-beta COMMIT " TOSTRING(APRILANN_COMMIT)
 #define APRILANN_COPYRIGHT "Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH"
 #define APRILANN_AUTHORS   "F. Zamora-Martinez, S. España-Boquera, J. Gorbe-Moya, J. Pastor & A. Palacios"

------------------------------------------------------------------------------

2013/09/04: Finally, here is the luaconf.h section for LUA_PATH and LUA_CPATH:

#define LUA_CDIR_32     LUA_ROOT "lib/i386-linux-gnu/lua/" LUA_VDIR
#define LUA_CDIR_64     LUA_ROOT "lib/x86_64-linux-gnu/lua/" LUA_VDIR
#define LUA_PATH_DEFAULT  \
  LUA_LDIR"?.lua;"  LUA_LDIR"?/init.lua;"		\
  LUA_CDIR"?.lua;"  LUA_CDIR"?/init.lua;" "./?.lua"
#define LUA_CPATH_DEFAULT				\
  LUA_CDIR"?.so;" LUA_CDIR"loadall.so;"			\
  LUA_CDIR_32"?.so;" LUA_CDIR_32"loadall.so;"		\
  LUA_CDIR_64"?.so;" LUA_CDIR_64"loadall.so;"		\
  "./?.so;"
#endif			/* } */

----------------------------------------------------------------------------

2013/09/03: Modified luaconf.h:

- #define LUA_ROOT	"/usr/local/"
+ #define LUA_ROOT	"/usr/"

----------------------------------------------------------------------------

2013/08/28: Modified Lua makefile, added this:

UNAME = `uname`
PLAT= DetectOs
DetectOs:
	-@make $(UNAME)
Linux: linux
Darwin: macosx

----------------------------------------------------------------------------

2013/06/08: Lua 5.2.2 update

- Changes of Lua 5.2.2

diff --git a/lua/lua-5.2.2/src/liolib.c b/lua/lua-5.2.2/src/liolib.c
index 3f80db1..d49d1db 100644
--- a/lua/lua-5.2.2/src/liolib.c
+++ b/lua/lua-5.2.2/src/liolib.c
@@ -46,40 +46,6 @@
 
 /*
 ** {======================================================
-** lua_popen spawns a new process connected to the current
-** one through the file streams.
-** =======================================================
-*/
-
-#if !defined(lua_popen)	/* { */
-
-#if defined(LUA_USE_POPEN)	/* { */
-
-#define lua_popen(L,c,m)	((void)L, fflush(NULL), popen(c,m))
-#define lua_pclose(L,file)	((void)L, pclose(file))
-
-#elif defined(LUA_WIN)		/* }{ */
-
-#define lua_popen(L,c,m)		((void)L, _popen(c,m))
-#define lua_pclose(L,file)		((void)L, _pclose(file))
-
-
-#else				/* }{ */
-
-#define lua_popen(L,c,m)		((void)((void)c, m),  \
-		luaL_error(L, LUA_QL("popen") " not supported"), (FILE*)0)
-#define lua_pclose(L,file)		((void)((void)L, file), -1)
-
-
-#endif				/* } */
-
-#endif			/* } */
-
-/* }====================================================== */
-
-
-/*
-** {======================================================
 ** lua_fseek/lua_ftell: configuration for longer offsets
 ** =======================================================
 */

diff --git a/lua/lua-5.2.2/src/llex.c b/lua/lua-5.2.2/src/llex.c
index 1a32e34..877dc34 100644
--- a/lua/lua-5.2.2/src/llex.c
+++ b/lua/lua-5.2.2/src/llex.c
@@ -33,7 +33,7 @@
 
 
 /* ORDER RESERVED */
-static const char *const luaX_tokens [] = {
+const char *const luaX_tokens [] = {
     "and", "break", "do", "else", "elseif",
     "end", "false", "for", "function", "goto", "if",
     "in", "local", "nil", "not", "or", "repeat",

diff --git a/lua/lua-5.2.2/src/lua.h b/lua/lua-5.2.2/src/lua.h
index f6fe0d4..c675516 100644
--- a/lua/lua-5.2.2/src/lua.h
+++ b/lua/lua-5.2.2/src/lua.h
@@ -28,7 +28,7 @@
 
 #define APRILANN_VERSION_MAJOR "0"
 #define APRILANN_VERSION_MINOR "2"
-#define APRILANN_VERSION_RELEASE "0"
+#define APRILANN_VERSION_RELEASE "1"
 #define APRILANN_RELEASE   "April-ANN v" APRILANN_VERSION_MAJOR "." APRILANN_VERSION_MINOR "." APRILANN_VERSION_RELEASE "-beta"
 #define APRILANN_COPYRIGHT "Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH"
 #define APRILANN_AUTHORS   "F. Zamora-Martinez, S. España-Boquera, J. Gorbe-Moya, J. Pastor & A. Palacios"

diff --git a/lua/lua-5.2.2/src/lualib.h b/lua/lua-5.2.2/src/lualib.h
index 9fd126b..f5977cb 100644
--- a/lua/lua-5.2.2/src/lualib.h
+++ b/lua/lua-5.2.2/src/lualib.h
@@ -10,6 +10,38 @@
 
 #include "lua.h"
 
+/*
+** {======================================================
+** lua_popen spawns a new process connected to the current
+** one through the file streams.
+** =======================================================
+*/
+
+#if !defined(lua_popen)	/* { */
+
+#if defined(LUA_USE_POPEN)	/* { */
+
+#define lua_popen(L,c,m)	((void)L, fflush(NULL), popen(c,m))
+#define lua_pclose(L,file)	((void)L, pclose(file))
+
+#elif defined(LUA_WIN)		/* }{ */
+
+#define lua_popen(L,c,m)		((void)L, _popen(c,m))
+#define lua_pclose(L,file)		((void)L, _pclose(file))
+
+
+#else				/* }{ */
+
+#define lua_popen(L,c,m)		((void)((void)c, m),  \
+		luaL_error(L, LUA_QL("popen") " not supported"), (FILE*)0)
+#define lua_pclose(L,file)		((void)((void)L, file), -1)
+
+
+#endif				/* } */
+
+#endif			/* } */
+
+/* }====================================================== */
 
 
 LUAMOD_API int (luaopen_base) (lua_State *L);

- Changes for compilation of lstrip/lstrip.c with Lua 5.2.2

diff --git a/lua/lstrip/lstrip.c b/lua/lstrip/lstrip.c
index 715f3b6..b99fa2b 100644
--- a/lua/lstrip/lstrip.c
+++ b/lua/lstrip/lstrip.c
@@ -12,6 +12,8 @@
 #include <stdlib.h>
 #include <string.h>
 
+#define lua_open()  luaL_newstate()
+
 #define LUA_CORE
 
 #include "lua.h"
@@ -31,6 +33,8 @@ static const char* progname=PROGNAME;	/* actual program name */
 static int preserve=0;			/* preserve line breaks? */
 static int dump=0;			/* dump instead of stripping? */
 
+extern const char *const luaX_tokens [];
+
 static void fatal(const char* message)
 {
  fprintf(stderr,"%s: %s\n",progname,message);
@@ -184,14 +188,24 @@ static void dostrip(LexState *X)
  }
 }
 
-Proto *luaY_parser(lua_State *L, ZIO *z, Mbuffer *buff, const char *name)
+Closure *luaY_parser(lua_State *L, ZIO *z, Mbuffer *buff,
+		     Dyndata *dyd, const char *name, int firstchar)
 {
  LexState X;
  FuncState F;
+ Closure *cl = luaF_newLclosure(L, 1);  /* create main closure */
+ /* anchor closure (to avoid being collected) */
+ setclLvalue(L, L->top, cl);
+ incr_top(L);
  X.buff=buff;
- luaX_setinput(L,&X,z,luaS_new(L,name));
+ F.f = cl->l.p = luaF_newproto(L);
+ F.f->source = luaS_new(L, name);  /* create and anchor TString */
+ X.buff = buff;
+ X.dyd = dyd;
+ dyd->actvar.n = dyd->gt.n = dyd->label.n = 0;
+ luaX_setinput(L, &X, z, F.f->source, firstchar);
  X.fs=&F;
- X.fs->h=luaH_new(L,0,0);
+ X.fs->h=luaH_new(L);
  sethvalue2s(L,L->top,X.fs->h);
  incr_top(L);
  if (dump)
@@ -200,7 +214,7 @@ Proto *luaY_parser(lua_State *L, ZIO *z, Mbuffer *buff, const char *name)
   dodump(&X);
  }
  else dostrip(&X);
- return luaF_newproto(L);
+ return cl;  /* it's on the stack too */
 }
 
 static void strip(lua_State *L, const char *file)

-------------------------------------
-- CHANGES BEFORE LUA 5.2.2 UPDATE --
-------------------------------------

07/06/2013

diff --git a/lua/lua-5.1.4/src/lua.c b/lua/lua-5.1.4/src/lua.c
index a650dbd..164dc12 100644
--- a/lua/lua-5.1.4/src/lua.c
+++ b/lua/lua-5.1.4/src/lua.c
@@ -110,6 +110,7 @@ static int docall (lua_State *L, int narg, int clear) {
 
 static void print_version (void) {
   l_message(NULL, LUA_RELEASE "  " LUA_COPYRIGHT);
+  l_message(NULL, APRILANN_RELEASE "  " APRILANN_COPYRIGHT);
 }
 
 
diff --git a/lua/lua-5.1.4/src/lua.h b/lua/lua-5.1.4/src/lua.h
index e4bdfd3..8956ab3 100644
--- a/lua/lua-5.1.4/src/lua.h
+++ b/lua/lua-5.1.4/src/lua.h
@@ -22,6 +22,9 @@
 #define LUA_COPYRIGHT	"Copyright (C) 1994-2008 Lua.org, PUC-Rio"
 #define LUA_AUTHORS 	"R. Ierusalimschy, L. H. de Figueiredo & W. Celes"
 
+#define APRILANN_RELEASE   "April-ANN 0.1.1 beta"
+#define APRILANN_COPYRIGHT "Copyright (C) 2012-2013 April-ANN"
+#define APRILANN_AUTHORS   "F. Zamora-Martinez, S. España-Boquera, J. Gorbe-Moya, J. Pastor & A. Palacios"
 
 /* mark for precompiled code (`<esc>Lua') */
 #define	LUA_SIGNATURE	"\033Lua"


----------------------------------------------------------------------------

02/06/2013
-  lua.c:22 static lua_State *globalL = NULL;
+  lua.c:22 lua_State *globalL = NULL;

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


Cambios realizados a la version estándar de lua-5.1.1
-----------------------------------------------------

En src/lib/liolib.c, funcion read_chars, cambio para evitar que un read termine interrumpido por se�al y se entienda como un EOF

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

En el fichero src/lib/liolib.c: A�adida la funci�n 'pclose'
para usar con el interprete de Lua. Las l�neas a�adidas est�n
m�rcadas con comentarios. Buscar 'Formiga' en el comentario.




En el fichero luac.c situado en src/luac hemos

añadido en la línea 93:

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

y hemos cambiado la línea 182:

antes:

FILE* D=fopen(output,"wb");

ahora:

FILE* D= (output==NULL) ? stdout : fopen(output,"wb");

de esta manera se permite poner -o - para que saque el bytecode por salida estándar.

Hemos cambiado tres líneas en el fichero config:

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
