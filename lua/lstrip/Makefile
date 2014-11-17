# makefile for lstrip

# change these to reflect your Lua installation (Lua 5.3!)
LUA= $(LUA_DIR)
LUAINC= $(LUA)/src
LUALIB= $(LUA)/src
LUASRC= $(LUA)/src

# no need to change anything below here
CFLAGS= $(INCS) $(WARN) -O2 $G
WARN= -O2 -Wall -Wextra

INCS= -I$(LUAINC) -I$(LUASRC)
LIBS= -L$(LUALIB) -llua -lm

MYNAME= lstrip
MYLIB= $(MYNAME)
T= $(MYNAME)
OBJS= $(MYNAME).o llex.o
TEST= test.lua
LUAEXECDIR = $(LUA_DIR)/../bin

all:	test

test:	$T
	$T $(TEST)

$T:	$(OBJS)
	$(CC) -o $@ $(OBJS) $(LIBS)
	cp $(MYNAME) $(LUAEXECDIR)

llex.c:
	sed '/void luaX_next/i#include "proxy.c"' $(LUASRC)/$@ > $@

llex.o:	proxy.c

clean:
	-rm -f $(OBJS) core core.* a.out llex.c $(MYNAME)

# eof
