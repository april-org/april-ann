#!/bin/bash
. configure.sh
if [[ $CC == gcc ]]; then
    echo "**** GCC RELEASE-ATLAS ****"     &&
    make release-atlas                     &&
    echo "**** GCC TEST-DEBUG-ATLAS ****"  &&
    make test-debug-atlas                  &&
    echo "**** GCC RELEASE-NO-OMP ****"    &&
    make release-no-omp                    &&
    echo "**** GCC TEST-DEBUG-NO-OMP ****" &&
    make test-debug-no-omp
elif [[ $CC == clang ]]; then
    echo "**** CLANG RELEASE-NO-OMP ****"     &&
    make release-no-omp                       &&
    echo "**** CLANG TEST-DEBUG-NO-OMP ****"  &&
    make test-debug-no-omp
else
    echo "Unknown variable CC=$CC"
    exit -1
fi
