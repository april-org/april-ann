#!/bin/bash
. configure.sh
UNAME=$(uname)
if [[ $UNAME == "Linux" ]]; then
    if [[ $CC == gcc ]]; then
	if [[ $OMP == "yes" ]]; then
            ## echo "**** GCC RELEASE-ATLAS ****"     &&
            ## make release-atlas                     &&
            echo "**** GCC TEST-DEBUG-ATLAS ****"  &&
            make test-debug-atlas
	else
            ## echo "**** GCC RELEASE-NO-OMP ****"    &&
            ## make release-no-omp                    &&
	    echo "**** GCC TEST-DEBUG-NO-OMP ****" &&
            make test-debug-no-omp
	fi
    elif [[ $CC == clang ]]; then
	if [[ $OMP == "no" ]]; then
            ## echo "**** CLANG RELEASE-NO-OMP ****"     &&
            ## make release-no-omp                       &&
            echo "**** CLANG TEST-DEBUG-NO-OMP ****"  &&
            make test-debug-no-omp
	else
	    echo "Unable to run OMP test with clang"
	    exit 10
	fi
    else
        echo "Unknown variable CC=$CC"
        exit 10
    fi
elif [[ $UNAME == "Darwin" ]]; then
    if [[ $CC == clang ]]; then
	if [[ $OMP == "no" ]]; then
            ## echo "**** CLANG RELEASE-HOMEBREW ****"     &&
            ## make release-macosx                       &&
            echo "**** CLANG TEST-DEBUG-HOMEBREW ****"  &&
            make test-debug-homebrew
	else
	    echo "Unable to run OMP test with clang"
	    exit 10
	fi
    else
        echo "Unknown variable CC=$CC"
        exit 10
    fi
else
    echo "Unknown UNAME=$UNAME"
    exit 10
fi
