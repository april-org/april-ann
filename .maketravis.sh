#!/bin/bash
. configure.sh
if [[ $CC == gcc ]]; then
    make release && make test && make release-pi
elif [[ $CC == clang ]]; then
    make release-no-omp
else
    echo "Unknown variable CC=$CC"
    exit -1
fi
