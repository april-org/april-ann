2013/06/08: Lua 5.2.2 update

- Changes for compilation of lstrip/lstrip.c with Lua 5.2.2

15 +#define lua_open()  luaL_newstate()
16 +
@@
36 +extern const char *const luaX_tokens [];
37 +
@@
187 -Proto *luaY_parser(lua_State *L, ZIO *z, Mbuffer *buff, const char *name)
191 +Closure *luaY_parser(lua_State *L, ZIO *z, Mbuffer *buff,
192 +         Dyndata *dyd, const char *name, int firstchar)
193 {
194   LexState X;
195   FuncState F;
196 + Closure *cl = luaF_newLclosure(L, 1);  /* create main closure */
197 + /* anchor closure (to avoid being collected) */
198 + setclLvalue(L, L->top, cl);
199 + incr_top(L);
200   X.buff=buff;
201 - luaX_setinput(L,&X,z,luaS_new(L,name));
202 + F.f = cl->l.p = luaF_newproto(L);
203 + F.f->source = luaS_new(L, name);  /* create and anchor TString */
204 + X.buff = buff;
205 + X.dyd = dyd;
206 + dyd->actvar.n = dyd->gt.n = dyd->label.n = 0;
207 + luaX_setinput(L, &X, z, F.f->source, firstchar);
208   X.fs=&F;
209 - X.fs->h=luaH_new(L,0,0);
210 + X.fs->h=luaH_new(L);
@@
203 - return luaF_newproto(L);
217 + return cl;  /* it's on the stack too */
@@

- Changes at lua-5.2.2/src/liolib.c

49 -** lua_popen spawns a new process connected to the current
50 -** one through the file streams.
51 -** =======================================================
52 -*/
53 -
54 -#if !defined(lua_popen)  /* { */
55 -
56 -#if defined(LUA_USE_POPEN)  /* { */
57 -
58 -#define lua_popen(L,c,m)  ((void)L, fflush(NULL), popen(c,m))
59 -#define lua_pclose(L,file)  ((void)L, pclose(file))
60 -
61 -#elif defined(LUA_WIN)    /* }{ */
62 -
63 -#define lua_popen(L,c,m)    ((void)L, _popen(c,m))
64 -#define lua_pclose(L,file)    ((void)L, _pclose(file))
65 -
66 -
67 -#else        /* }{ */
68 -
69 -#define lua_popen(L,c,m)    ((void)((void)c, m),  \
70 -    luaL_error(L, LUA_QL("popen") " not supported"), (FILE*)0)
71 -#define lua_pclose(L,file)    ((void)((void)L, file), -1)
72 -
73 -
74 -#endif        /* } */
75 -
76 -#endif      /* } */
77 -
78 -/* }====================================================== */
79 -
80 -
81 -/*
82 -** {======================================================

- Changes at lua-5.2.2/src/llex.c

36 - static const char *const luaX_tokens [] = {
36 +const char *const luaX_tokens [] = {

- Changes at lua-5.2.2/src/liolib.h

13 +/*
14 +** {======================================================
15 +** lua_popen spawns a new process connected to the current
16 +** one through the file streams.
17 +** =======================================================
18 +*/
19 +
20 +#if !defined(lua_popen)  /* { */
21 +
22 +#if defined(LUA_USE_POPEN)  /* { */
23 +
24 +#define lua_popen(L,c,m)  ((void)L, fflush(NULL), popen(c,m))
25 +#define lua_pclose(L,file)  ((void)L, pclose(file))
26 +
27 +#elif defined(LUA_WIN)    /* }{ */
28 +
29 +#define lua_popen(L,c,m)    ((void)L, _popen(c,m))
30 +#define lua_pclose(L,file)    ((void)L, _pclose(file))
31 +
32 +
33 +#else        /* }{ */
34 +
35 +#define lua_popen(L,c,m)    ((void)((void)c, m),  \
36 +    luaL_error(L, LUA_QL("popen") " not supported"), (FILE*)0)
37 +#define lua_pclose(L,file)    ((void)((void)L, file), -1)
38 +
39 +
40 +#endif        /* } */
41 +
42 +#endif      /* } */
43 +
44 +/* }====================================================== */

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
+#define APRILANN_AUTHORS   "F. Zamora-Martinez, S. EspaÃ±a-Boquera, J. Gorbe-Moya, J. Pastor & A. Palacios"
 
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


Cambios realizados a la version estÃ¡ndar de lua-5.1.1
-----------------------------------------------------

En src/lib/liolib.c, funcion read_chars, cambio para evitar que un read termine interrumpido por señal y se entienda como un EOF

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

En el fichero src/lib/liolib.c: Añadida la función 'pclose'
para usar con el interprete de Lua. Las líneas añadidas están
márcadas con comentarios. Buscar 'Formiga' en el comentario.




En el fichero luac.c situado en src/luac hemos

aÃ±adido en la lÃ­nea 93:

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

y hemos cambiado la lÃ­nea 182:

antes:

FILE* D=fopen(output,"wb");

ahora:

FILE* D= (output==NULL) ? stdout : fopen(output,"wb");

de esta manera se permite poner -o - para que saque el bytecode por salida estÃ¡ndar.

Hemos cambiado tres lÃ­neas en el fichero config:

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

