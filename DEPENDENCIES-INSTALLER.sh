#!/bin/bash
UNAME=$(uname)
echo "System: $UNAME"
if [ $UNAME = "Linux" ]; then
    if [ $(which apt-get) ]; then
        ubuntu_release=$(lsb_release -r | cut -f 2)
        if [[ $? -ne 0 ]]; then
            echo "Unable to call lsb_release command. This script only works in Ubuntu"
            exit 10
        fi
        if [[ $ubuntu_release == "12.04" ]]; then
	    packages="gfortran cmake pkg-config libz-dev libreadline-dev libblas-dev libatlas-dev libatlas-base-dev libpng12-dev libtiff-dev liblua5.2-dev libncurses5 libncurses5-dev liblapack-dev libzip-dev"
	    if ! dpkg -l $packages > /dev/null; then
		sudo apt-get -qq update
		sudo apt-get install -qq $packages
		if ! ldconfig -p | grep liblapacke > /dev/null; then
                    cwd=$(pwd)
                    cd /tmp/ &&
                    wget http://www.netlib.org/lapack/lapack-3.5.0.tgz &&
                    tar zxvf lapack-3.5.0.tgz &&
                    cd lapack-3.5.0 &&
                    cmake -DBUILD_SHARED_LIBS=1 -DLAPACKE=1 -DCMAKE_INSTALL_PREFIX=/usr . &&
                    make &&
                    sudo make install
                    if [[ $? -ne 0 ]]; then
			echo "Error installing dependencies"
			exit 10
                    fi
                    cd $cwd
		fi
	    fi
        else
	    packages="gfortran pkg-config libz-dev libreadline-dev libblas-dev libatlas-dev libatlas-base-dev libpng12-dev libtiff-dev liblua5.2-dev libncurses5 libncurses5-dev liblapacke-dev libzip-dev"
	    if ! dpkg -l $packages > /dev/null; then
		sudo apt-get -qq update
		sudo apt-get install -qq gfortran pkg-config libz-dev libreadline-dev libblas-dev libatlas-dev libatlas-base-dev libpng12-dev libtiff-dev liblua5.2-dev libncurses5 libncurses5-dev liblapacke-dev libzip-dev
		if [[ $? -ne 0 ]]; then
                    echo "Error installing dependencies, only works with ubuntu >= 12.04"
                    exit 10
		fi
	    fi
        fi
        dest=$(tempfile)
        echo -e '#include "lua5.2-deb-multiarch.h"\nint main() { return 0; }\n' > $dest.c
        multiarch=lua/include/lua5.2-deb-multiarch.h
        multiarch2=lua/lua-5.2.2/src/lua5.2-deb-multiarch.h
        if ! gcc -c $dest.c 2> /dev/null; then
            echo "Not found lua5.2-deb-multiarch.h, generating a default one in $multiarch, please, check the content"
            echo -e "#ifndef _LUA_DEB_MULTIARCH_\n#define _LUA_DEB_MULTIARCH_\n#define DEB_HOST_MULTIARCH \""$(arch)"-linux-gnu\"\n#endif" > $multiarch
            echo -e "#ifndef _LUA_DEB_MULTIARCH_\n#define _LUA_DEB_MULTIARCH_\n#define DEB_HOST_MULTIARCH \""$(arch)"-linux-gnu\"\n#endif" > $multiarch2
        fi
        if [ ! -z $dest ]; then
            rm -f $dest*
            rm -f $(basename $dest).o
        fi
    else
        echo "Error, impossible to install dependencies, this script only works with apt-get"
        exit 10
    fi
elif [ $UNAME = "Darwin" ]; then
    if [ $(which port) ]; then
        if ! sudo port install zlib readline libpng tiff findutils pkgconfig lua libzip; then
            echo "ERROR INSTALLING DEPENDENCIES USING MACPORTS"
            exit 10
        fi
    elif [ $(which brew) ]; then
        brew update
        # If the packages are installed, brew is returning != 0 error code
        #if ! brew install lzlib readline libpng libtiff findutils pkgconfig; then
        #    echo "ERROR INSTALLING DEPENDENCIES USING HOMEBREW"
        #    exit 10
        #fi
        brew install lzlib readline libpng libtiff findutils pkgconfig libzip
        ###################################################################
        # THIS UGLY HACK IS BECAUSE LUA52 IS NOT AT DEFAULT HOMEBREW REPO #
        brew tap homebrew/versions
        # If the packages are installed, brew is returning != 0 error code
        #if ! brew install lua52; then
        #    echo "ERROR INSTALLING DEPENDENCIES USING HOMEBREW"
        #    exit 10
        #fi
        brew install lua52
        brew link lua52
        ###################################################################
    else
        echo "Error, impossible to install dependencies, this scripts needs MacPorts or Homebrew"
        exit 10
    fi

    echo "This script will change the default system BSD find by GNU find"
    echo "BSD find will be renamed as bfind"
    if [ ! -e /usr/find/bfind ]; then
        sudo mv /usr/bin/find /usr/bin/bfind
    else
        sudo rm -f /usr/bin/find
    fi
    sudo ln -s $(which gfind) /usr/bin/find

else
    echo "Error, impossible to install dependencies, not recognized system: $UNAME"
    exit 10
fi
exit 0
