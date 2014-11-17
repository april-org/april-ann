/*
* lstrip.c
* compress Lua programs by removing comments and whitespaces,
* optionally preserving line breaks (for error messages).
* 08 Oct 2012 10:50:57
* Luiz Henrique de Figueiredo <lhf@tecgraf.puc-rio.br>
* This code is hereby placed in the public domain.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lua.h"
#include "lauxlib.h"

static const char* progname="lstrip";	/* actual program name */
int lstrip_options=0;			/* read by proxy.c */

static void fatal(const char* message)
{
 fprintf(stderr,"%s: %s\n",progname,message);
 exit(EXIT_FAILURE);
}

static void strip(lua_State *L, const char *file)
{
 if (file!=NULL && strcmp(file,"-")==0) file=NULL;
 if (luaL_loadfile(L,file)!=0) fatal(lua_tostring(L,-1));
 lua_settop(L,0);
}

int main(int argc, char* argv[])
{
  int preserve=0;			/* preserve line breaks? */
 int dump=0;				/* dump instead of stripping? */
 lua_State *L=luaL_newstate();
 (void)(argc); // remove compiler warning
 if (argv[0]!=NULL && *argv[0]!=0) progname=argv[0];
 if (L==NULL) fatal("not enough memory for state");
 while (*++argv!=NULL && strcmp(*argv,"-p")==0) preserve++;
 --argv;
 while (*++argv!=NULL && strcmp(*argv,"-d")==0) dump++;
 if (dump) lstrip_options=-1; else lstrip_options=preserve;
 if (*argv==NULL)
  strip(L,NULL);
 else
  while (*argv) strip(L,*argv++);
 lua_close(L);
 return EXIT_SUCCESS;
}
