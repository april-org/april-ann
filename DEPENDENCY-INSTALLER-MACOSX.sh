#!/bin/bash
port install zlib readline libpng tiff findutils
echo "This script will change the default system BSD find by GNU find"
echo "BSD find will be renamed as bfind"
mv /usr/bin/find /usr/find/bfind
ln -s `which gfind` /usr/bin/find
