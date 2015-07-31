folder=`ls luapkg`
if [ -z "$folder" ]; then
    echo "Cloning submodules"
    git submodule init
    git submodule update
fi
git submodule foreach git pull origin master
if [ -z $APRILANN_CONFIGURED ]; then
    export APRILANN_CONFIGURED=1
#export LANG=""
    export PATH="`pwd`"/lua/bin:`pwd`/bin:$PATH
#export LUA_PATH="`pwd`""/luapkg/?.lua;?"
fi
if [ -z $LUA_HISTSIZE ]; then
    export LUA_HISTSIZE=500
fi
if [ -z $LUA_HISTORY ]; then
    export LUA_HISTORY=~/.lua_history
fi
export LUA_DIR="`pwd`"/lua/lua-5.2.2
(make -C $LUA_DIR            &&
 make -C $LUA_DIR install    &&
 make -C "`pwd`"/lua/lstrip) ||
(echo "Error building Lua!!!" && exit -1)
if [ -z $APRIL_EXEC ]; then
    export APRIL_EXEC=`pwd`/bin/april-ann
fi
if [ -z $APRIL_TOOLS_DIR ]; then
    export APRIL_TOOLS_DIR=`pwd`/tools/
fi
