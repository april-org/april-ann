export LUA_DIR="`pwd`"/lua/lua-5.2.2
make -C $LUA_DIR
make -C $LUA_DIR install
make -C "`pwd`"/lua/lstrip
#export LANG=""
export PATH="`pwd`"/lua/bin:`pwd`/bin:$PATH
export LUA_PATH="`pwd`""/binding/?.lua;?"
if [ -z $APRIL_EXEC ]; then
    export APRIL_EXEC=`pwd`/bin/april-ann
fi
if [ -z $APRIL_TOOLS_DIR ]; then
    export APRIL_TOOLS_DIR=`pwd`/tools/
fi
