/*
* proxy.c
* token filter for lstrip for Lua 5.3
* Luiz Henrique de Figueiredo <lhf@tecgraf.puc-rio.br>
* 03 Jul 2014 11:31:27
* This code is hereby placed in the public domain.
*/

#include <ctype.h>
#include <stdio.h>

extern int lstrip_options;
#define preserve lstrip_options

static void quote(const TString *ts)
{
 const char* s=getstr(ts);
 size_t i,n=ts->len;
 printf("%c",'"');
 for (i=0; i<n; i++)
 {
  int c=(int)(unsigned char)s[i];
  switch (c)
  {
   case '"':  printf("\\\""); break;
   case '\\': printf("\\\\"); break;
   case '\a': printf("\\a"); break;
   case '\b': printf("\\b"); break;
   case '\f': printf("\\f"); break;
   case '\n': printf("\\n"); break;
   case '\r': printf("\\r"); break;
   case '\t': printf("\\t"); break;
   case '\v': printf("\\v"); break;
   default:	if (isprint(c))
   			printf("%c",c);
		else
			printf("\\%03d",c);
  }
 }
 printf("%c",'"');
}

#define	TK_NUMBER	TK_FLT

static int dodump(LexState *X, SemInfo *seminfo)
{
 printf("0\t<file>\t%s\n",getstr(X->source));
 for (;;)
 {
  int t=llex(X,seminfo);
  if (t==TK_INT) t=TK_NUMBER;
  printf("%d\t",X->linenumber);
  if (t<FIRST_RESERVED)
   printf("%c",t);
  else
   printf("%s",luaX_tokens[t-FIRST_RESERVED]);
  printf("\t");
  switch (t)
  {
   case TK_EOS:
    printf("\n");
    return TK_EOS;
   case TK_STRING:
    quote(seminfo->ts);
    break;
   case TK_NAME:
    printf("%s",getstr(seminfo->ts));
    break;
   case TK_NUMBER:
    printf("%s",X->buff->buffer);
    break;
  }
  printf("\n");
 }
}

#define pair(a,b)	(1024*(a)+(b))

static int clash[]= {
	pair('-', '-'),
	pair('[', '='),
	pair('[', '['),
	pair('[', TK_EQ),
	pair('=', '='),
	pair('=', TK_EQ),
	pair('<', '='),
	pair('<', TK_EQ),
	pair('<', '<'),
	pair('<', TK_LE),
	pair('<', TK_SHL),
	pair('>', '='),
	pair('>', TK_EQ),
	pair('>', '>'),
	pair('>', TK_GE),
	pair('>', TK_SHR),
	pair('/', '/'),
	pair('/', TK_IDIV),
	pair('~', '='),
	pair('~', TK_EQ),
	pair(':', ':'),
	pair(':', TK_DBCOLON),
	pair('.', '.'),
	pair('.', TK_CONCAT),
	pair('.', TK_DOTS),
	pair('.', TK_NUMBER),
	pair(TK_CONCAT, '.'),
	pair(TK_CONCAT, TK_CONCAT),
	pair(TK_CONCAT, TK_DOTS),
	pair(TK_CONCAT, TK_NUMBER),
	pair(TK_NAME, TK_NAME),
	pair(TK_NAME, TK_NUMBER),
	pair(TK_NUMBER, '.'),
	pair(TK_NUMBER, TK_CONCAT),
	pair(TK_NUMBER, TK_DOTS),
	pair(TK_NUMBER, TK_NAME),
	pair(TK_NUMBER, TK_NUMBER),
	0
};

static int space(int a, int b)
{
 int i,c;
 if (a>=FIRST_RESERVED && a<=TK_WHILE) a=TK_NAME;
 if (b>=FIRST_RESERVED && b<=TK_WHILE) b=TK_NAME;
 c=pair(a,b);
 for (i=0; clash[i]!=0; i++)
  if (c==clash[i]) return 1;
 return 0;
}

static int dostrip(LexState *X, SemInfo *seminfo)
{
 int ln=1;
 int lt=0;
 for (;;)
 {
  int t=llex(X,seminfo);
  if (t==TK_INT) t=TK_NUMBER;
  if (preserve)
  {
   if (X->linenumber!=ln)
   {
    if (preserve>1)
     while (X->linenumber!=ln++) printf("\n");
    else if (lt!=0)
     printf("\n");
    ln=X->linenumber;
    lt=0;
   }
  }
  if (space(lt,t)) printf(" ");
  switch (t)
  {
   case TK_EOS:
    return TK_EOS;
   case TK_STRING:
    quote(seminfo->ts);
    break;
   case TK_NAME:
    printf("%s",getstr(seminfo->ts));
    break;
   case TK_NUMBER:
    printf("%s",X->buff->buffer);
    break;
   default:
    if (t<FIRST_RESERVED)
     printf("%c",t);
    else
     printf("%s",luaX_tokens[t-FIRST_RESERVED]);
    break;
  }
  lt=t;
  if (preserve>2) printf("\n");
 }
}

static int nexttoken(LexState *X, SemInfo *seminfo)
{
 if (preserve<0)
  return dodump(X,seminfo);
 else
  return dostrip(X,seminfo);
}

#define llex nexttoken
