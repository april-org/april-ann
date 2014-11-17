local F={
	[0]={
		"frexp",
		"fmax",
		"fmin",
		"modf",
		"ldexp",
	},
	[1]={
		"abs",
		"acos",
		"acosh",
		"asin",
		"asinh",
		"atan",
		"atanh",
		"cbrt",
		"ceil",
		"cos",
		"cosh",
		"erf",
		"erfc",
		"exp",
		"exp2",
		"expm1",
		"floor",
		"gamma",
		"lgamma",
		"log",
		"log10",
		"log1p",
		"log2",
		"logb",
		"nearbyint",
		"round",
		"sin",
		"sinh",
		"sqrt",
		"tan",
		"tanh",
		"trunc",
	},
	[2]={
		"atan2",
		"copysign",
		"fdim",
		"fmod",
		"hypot",
		"nextafter",
		"pow",
		"remainder",
		"scalbn",
	},
	[3]={
		"fma",
	},
	[4]={
		"isfinite",
		"isinf",
		"isnan",
		"isnormal",
	},
	[5]={
		"deg",
		"rad",
		"j0",
		"j1",
		"y0",
		"y1",
	},
	[6]={
		"jn",
		"yn",
	},
}

local T={
	[0]=[[
static int LX(lua_State *L)
{
 lua_pushnumber(L,l_mathop(X)(A));
 return 1;
}

]],
}
T[1]=T[0]:gsub("A","A(1)")
T[2]=T[0]:gsub("A","A(1),A(2)")
T[3]=T[0]:gsub("A","A(1),A(2),A(3)")
T[0]=T[0]:gsub("l_mathop.X.","X")
T[4]=T[0]:gsub("A","A(1)"):gsub("number","boolean")
T[5]=T[0]:gsub("A","A(1)")
T[6]=T[0]:gsub("A","I(1),A(2)")
T[0]=""

local K={}
local n=0
for i=0,#F do
	for j=1,#F[i] do
		local k=F[i][j]
		n=n+1
		K[n]=k
		K[k]=i
	end
end
table.sort(K)

local write=io.write
for i=1,#K do
	local k=K[i]
	local n=K[k]
	local t=T[n]:gsub("X",k)
	write(t)
end

write[[
static const luaL_Reg R[] =
{
]]
local T=[[
	{ "X",	LX },
]]
for i=1,#K do
	local k=K[i]
	local t=T:gsub("X",k)
	write(t)
end
write[[
	{ NULL,	NULL }
};
]]

do return end

write[[

/*

]]
for i=1,#K do
	write(K[i],"\n")
end
write[[

*/
]]
