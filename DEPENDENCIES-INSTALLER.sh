#!/bin/bash
UNAME=`uname`
echo "System: $UNAME"
if [ $UNAME = "Linux" ]; then
    if [ `which apt-get` ]; then
	apt-get install libz-dev libreadline-dev libblas-dev libatlas-dev libpng12-dev libtiff-dev liblua5.2-dev ||
	(echo "ERROR INSTALLING DEPENDENCIES" && exit 10)
	dest=$(tempfile)
	echo -e '#include "lua5.2-deb-multiarch.h"\nint main() { return 0; }\n' > $dest.c
	multiarch=lua/lua-5.2.2/src/lua5.2-deb-multiarch.h
	if ! gcc -c $dest.c 2> /dev/null; then
	    echo "Not found lua5.2-deb-multiarch.h, generating a default one in $multiarch, please, check the content"
	    echo -e "#ifndef _LUA_DEB_MULTIARCH_\n#define _LUA_DEB_MULTIARCH_\n#define DEB_HOST_MULTIARCH \""$(arch)"-linux-gnu\"\n#endif" > $multiarch
	fi
	rm -f $dest*
	rm -f $(basename $dest).o
    else
	echo "Error, impossible to install dependencies, this script only works with apt-get"
	exit 10
    fi
elif [ $UNAME = "Darwin" ]; then
    if [ `which port` ]; then
	port install zlib readline libpng tiff findutils ||
	(echo "ERROR INSTALLING DEPENDENCIES" && exit 10)
	echo "This script will change the default system BSD find by GNU find"
	echo "BSD find will be renamed as bfind"
	if [ ! -e /usr/find/bfind ]; then
	    mv /usr/bin/find /usr/find/bfind
	else
	    rm -f /usr/bin/find
	fi
	ln -s `which gfind` /usr/bin/find
    else
	echo "Error, impossible to install dependencies, this scripts needs MacPorts"
	exit 10
    fi
else
    echo "Error, impossible to install dependencies, not recognized system: $UNAME"
    exit 10
fi
exit 0
