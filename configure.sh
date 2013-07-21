export LUA_DIR="`pwd`"/lua/lua-5.2.2
make -C $LUA_DIR
make -C $LUA_DIR install
make -C "`pwd`"/lua/lstrip
#export LANG=""
export PATH="`pwd`"/lua/bin:`pwd`/bin:$PATH
export LUA_PATH="`pwd`""/binding/?.lua;?"
