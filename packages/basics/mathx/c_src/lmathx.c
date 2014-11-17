/*
* lmathx.c
* C99 math functions for Lua
* Luiz Henrique de Figueiredo <lhf@tecgraf.puc-rio.br>
* 11 Jun 2014 22:48:52
* This code is hereby placed in the public domain.
*/

#define _GNU_SOURCE 1
#include <math.h>
#include <stdio.h>

#include "lua.h"
#include "lualib.h"
#include "lauxlib.h"

#define A(i)	luaL_checknumber(L,i)
#define I(i)	luaL_checkint(L,i)

#undef abs
#define	abs	fabs
#define	gamma	tgamma

#undef PI
#define PI	(l_mathop(3.141592653589793238462643383279502884))
#define	rad(x)	((x)*(PI/180.0))
#define	deg(x)	((x)*(180.0/PI))

static int Lfmax(lua_State *L)
{
 int i,n=lua_gettop(L);
 lua_Number m=A(1);
 for (i=2; i<=n; i++) m=fmax(m,A(i));
 lua_pushnumber(L,m);
 return 1;
}

static int Lfmin(lua_State *L)
{
 int i,n=lua_gettop(L);
 lua_Number m=A(1);
 for (i=2; i<=n; i++) m=fmin(m,A(i));
 lua_pushnumber(L,m);
 return 1;
}

static int Lfrexp(lua_State *L)
{
 int e;
 lua_pushnumber(L,l_mathop(frexp)(A(1),&e));
 lua_pushinteger(L,e);
 return 2;
}

static int Lldexp(lua_State *L)
{
 lua_pushnumber(L,l_mathop(ldexp)(A(1),I(2)));
 return 1;
}

static int Lmodf(lua_State *L)
{
 lua_Number ip;
 lua_Number fp=l_mathop(modf)(A(1),&ip);
 lua_pushnumber(L,ip);
 lua_pushnumber(L,fp);
 return 2;
}

static int Labs(lua_State *L)
{
 lua_pushnumber(L,l_mathop(abs)(A(1)));
 return 1;
}

static int Lacos(lua_State *L)
{
 lua_pushnumber(L,l_mathop(acos)(A(1)));
 return 1;
}

static int Lacosh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(acosh)(A(1)));
 return 1;
}

static int Lasin(lua_State *L)
{
 lua_pushnumber(L,l_mathop(asin)(A(1)));
 return 1;
}

static int Lasinh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(asinh)(A(1)));
 return 1;
}

static int Latan(lua_State *L)
{
 lua_pushnumber(L,l_mathop(atan)(A(1)));
 return 1;
}

static int Latan2(lua_State *L)
{
 lua_pushnumber(L,l_mathop(atan2)(A(1),A(2)));
 return 1;
}

static int Latanh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(atanh)(A(1)));
 return 1;
}

static int Lcbrt(lua_State *L)
{
 lua_pushnumber(L,l_mathop(cbrt)(A(1)));
 return 1;
}

static int Lceil(lua_State *L)
{
 lua_pushnumber(L,l_mathop(ceil)(A(1)));
 return 1;
}

static int Lcopysign(lua_State *L)
{
 lua_pushnumber(L,l_mathop(copysign)(A(1),A(2)));
 return 1;
}

static int Lcos(lua_State *L)
{
 lua_pushnumber(L,l_mathop(cos)(A(1)));
 return 1;
}

static int Lcosh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(cosh)(A(1)));
 return 1;
}

static int Ldeg(lua_State *L)
{
 lua_pushnumber(L,deg(A(1)));
 return 1;
}

static int Lerf(lua_State *L)
{
 lua_pushnumber(L,l_mathop(erf)(A(1)));
 return 1;
}

static int Lerfc(lua_State *L)
{
 lua_pushnumber(L,l_mathop(erfc)(A(1)));
 return 1;
}

static int Lexp(lua_State *L)
{
 lua_pushnumber(L,l_mathop(exp)(A(1)));
 return 1;
}

static int Lexp2(lua_State *L)
{
 lua_pushnumber(L,l_mathop(exp2)(A(1)));
 return 1;
}

static int Lexpm1(lua_State *L)
{
 lua_pushnumber(L,l_mathop(expm1)(A(1)));
 return 1;
}

static int Lfdim(lua_State *L)
{
 lua_pushnumber(L,l_mathop(fdim)(A(1),A(2)));
 return 1;
}

static int Lfloor(lua_State *L)
{
 lua_pushnumber(L,l_mathop(floor)(A(1)));
 return 1;
}

static int Lfma(lua_State *L)
{
 lua_pushnumber(L,l_mathop(fma)(A(1),A(2),A(3)));
 return 1;
}

static int Lfmod(lua_State *L)
{
 lua_pushnumber(L,l_mathop(fmod)(A(1),A(2)));
 return 1;
}

static int Lgamma(lua_State *L)
{
 lua_pushnumber(L,l_mathop(gamma)(A(1)));
 return 1;
}

static int Lhypot(lua_State *L)
{
 lua_pushnumber(L,l_mathop(hypot)(A(1),A(2)));
 return 1;
}

static int Lisfinite(lua_State *L)
{
 lua_pushboolean(L,isfinite(A(1)));
 return 1;
}

static int Lisinf(lua_State *L)
{
 lua_pushboolean(L,isinf(A(1)));
 return 1;
}

static int Lisnan(lua_State *L)
{
 lua_pushboolean(L,isnan(A(1)));
 return 1;
}

static int Lisnormal(lua_State *L)
{
 lua_pushboolean(L,isnormal(A(1)));
 return 1;
}

static int Lj0(lua_State *L)
{
 lua_pushnumber(L,j0(A(1)));
 return 1;
}

static int Lj1(lua_State *L)
{
 lua_pushnumber(L,j1(A(1)));
 return 1;
}

static int Ljn(lua_State *L)
{
 lua_pushnumber(L,jn(I(1),A(2)));
 return 1;
}

static int Llgamma(lua_State *L)
{
 lua_pushnumber(L,l_mathop(lgamma)(A(1)));
 return 1;
}

static int Llog(lua_State *L)
{
 lua_pushnumber(L,l_mathop(log)(A(1)));
 return 1;
}

static int Llog10(lua_State *L)
{
 lua_pushnumber(L,l_mathop(log10)(A(1)));
 return 1;
}

static int Llog1p(lua_State *L)
{
 lua_pushnumber(L,l_mathop(log1p)(A(1)));
 return 1;
}

static int Llog2(lua_State *L)
{
 lua_pushnumber(L,l_mathop(log2)(A(1)));
 return 1;
}

static int Llogb(lua_State *L)
{
 lua_pushnumber(L,l_mathop(logb)(A(1)));
 return 1;
}

static int Lnearbyint(lua_State *L)
{
 lua_pushnumber(L,l_mathop(nearbyint)(A(1)));
 return 1;
}

static int Lnextafter(lua_State *L)
{
 lua_pushnumber(L,l_mathop(nextafter)(A(1),A(2)));
 return 1;
}

static int Lpow(lua_State *L)
{
 lua_pushnumber(L,l_mathop(pow)(A(1),A(2)));
 return 1;
}

static int Lrad(lua_State *L)
{
 lua_pushnumber(L,rad(A(1)));
 return 1;
}

static int Lremainder(lua_State *L)
{
 lua_pushnumber(L,l_mathop(remainder)(A(1),A(2)));
 return 1;
}

static int Lround(lua_State *L)
{
 lua_pushnumber(L,l_mathop(round)(A(1)));
 return 1;
}

static int Lscalbn(lua_State *L)
{
 lua_pushnumber(L,l_mathop(scalbn)(A(1),A(2)));
 return 1;
}

static int Lsin(lua_State *L)
{
 lua_pushnumber(L,l_mathop(sin)(A(1)));
 return 1;
}

static int Lsinh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(sinh)(A(1)));
 return 1;
}

static int Lsqrt(lua_State *L)
{
 lua_pushnumber(L,l_mathop(sqrt)(A(1)));
 return 1;
}

static int Ltan(lua_State *L)
{
 lua_pushnumber(L,l_mathop(tan)(A(1)));
 return 1;
}

static int Ltanh(lua_State *L)
{
 lua_pushnumber(L,l_mathop(tanh)(A(1)));
 return 1;
}

static int Ltrunc(lua_State *L)
{
 lua_pushnumber(L,l_mathop(trunc)(A(1)));
 return 1;
}

static int Ly0(lua_State *L)
{
 lua_pushnumber(L,y0(A(1)));
 return 1;
}

static int Ly1(lua_State *L)
{
 lua_pushnumber(L,y1(A(1)));
 return 1;
}

static int Lyn(lua_State *L)
{
 lua_pushnumber(L,yn(I(1),A(2)));
 return 1;
}

static const luaL_Reg R[] =
{
	{ "abs",	Labs },
	{ "acos",	Lacos },
	{ "acosh",	Lacosh },
	{ "asin",	Lasin },
	{ "asinh",	Lasinh },
	{ "atan",	Latan },
	{ "atan2",	Latan2 },
	{ "atanh",	Latanh },
	{ "cbrt",	Lcbrt },
	{ "ceil",	Lceil },
	{ "copysign",	Lcopysign },
	{ "cos",	Lcos },
	{ "cosh",	Lcosh },
	{ "deg",	Ldeg },
	{ "erf",	Lerf },
	{ "erfc",	Lerfc },
	{ "exp",	Lexp },
	{ "exp2",	Lexp2 },
	{ "expm1",	Lexpm1 },
	{ "fdim",	Lfdim },
	{ "floor",	Lfloor },
	{ "fma",	Lfma },
	{ "fmax",	Lfmax },
	{ "fmin",	Lfmin },
	{ "fmod",	Lfmod },
	{ "frexp",	Lfrexp },
	{ "gamma",	Lgamma },
	{ "hypot",	Lhypot },
	{ "isfinite",	Lisfinite },
	{ "isinf",	Lisinf },
	{ "isnan",	Lisnan },
	{ "isnormal",	Lisnormal },
	{ "j0",	Lj0 },
	{ "j1",	Lj1 },
	{ "jn",	Ljn },
	{ "ldexp",	Lldexp },
	{ "lgamma",	Llgamma },
	{ "log",	Llog },
	{ "log10",	Llog10 },
	{ "log1p",	Llog1p },
	{ "log2",	Llog2 },
	{ "logb",	Llogb },
	{ "modf",	Lmodf },
	{ "nearbyint",	Lnearbyint },
	{ "nextafter",	Lnextafter },
	{ "pow",	Lpow },
	{ "rad",	Lrad },
	{ "remainder",	Lremainder },
	{ "round",	Lround },
	{ "scalbn",	Lscalbn },
	{ "sin",	Lsin },
	{ "sinh",	Lsinh },
	{ "sqrt",	Lsqrt },
	{ "tan",	Ltan },
	{ "tanh",	Ltanh },
	{ "trunc",	Ltrunc },
	{ "y0",	Ly0 },
	{ "y1",	Ly1 },
	{ "yn",	Lyn },
	{ NULL,	NULL }
};

LUALIB_API int luaopen_mathx(lua_State *L)
{
 if (!lua_getglobal(L,LUA_MATHLIBNAME)) lua_newtable(L);
 luaL_setfuncs(L,R,0);
 lua_pushnumber(L,PI);		lua_setfield(L,-2,"pi");
 lua_pushnumber(L,HUGE_VAL);	lua_setfield(L,-2,"huge");
 lua_pushnumber(L,INFINITY);	lua_setfield(L,-2,"inf");
 lua_pushnumber(L,NAN);		lua_setfield(L,-2,"nan");
 return 1;
}
