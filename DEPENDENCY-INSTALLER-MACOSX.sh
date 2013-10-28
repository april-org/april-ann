#!/bin/bash
port install zlib readline libpng tiff findutils
echo "This script will change the default system BSD find by GNU find"
echo "BSD find will be renamed as bfind"
if [ ! -e /usr/find/bfind ]; then
    mv /usr/bin/find /usr/find/bfind
else
    rm -f /usr/bin/find
fi
ln -s `which gfind` /usr/bin/find
