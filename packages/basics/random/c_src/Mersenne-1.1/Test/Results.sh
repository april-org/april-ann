#!/bin/sh

# Results.sh
# Examine the output from tests of Mersenne Twister random number generators
# Explain the meaning in plain English

rm -f Bug.txt

echo

echo "Examining performance of Wagner's MersenneTwister.h C++ class..."

# Extract the various distribution series from each generator's output

for NAME in *.out; do
	BASE=`basename $NAME .out`
	grep -A 200 "integer generation:" $NAME | tail -200 > $BASE.integer.tmp
	grep -A 200 -F "[0,1) generation:" $NAME | tail -200 > $BASE.exclusive.tmp
	grep -A 400 -F "[0,1] generation:" $NAME | tail -400 > $BASE.inclusive.tmp
done

# Compare the output to the inventors' reference output

rm -f *.reference.tmp
for NAME in *.out; do
	BASE=`basename $NAME .out`
	diff --brief $BASE.integer.tmp Reference.integer.tmp >> $BASE.reference.tmp
	diff --brief $BASE.exclusive.tmp Reference.exclusive.tmp >> $BASE.reference.tmp
	diff --brief $BASE.inclusive.tmp Reference.inclusive.tmp >> $BASE.reference.tmp
done

# Look for an available awk command

if which awk &> /dev/null
then
	AWK=awk
elif which gawk &> /dev/null
then
	AWK=gawk
elif which nawk &> /dev/null
then
	AWK=nawk
fi

# Report comparison test results

if [ $AWK ]; then
	echo
	echo "Comparison of integer generation rate with other versions:"
	
	printf "  This version                 "
	if [ -s Wagner.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Wagner.out
		if [ -s Wagner.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Inventors' version           "
	if [ -s Original.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Original.out
		if [ -s Original.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Bedaux's version             "
	if [ -s Bedaux.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Bedaux.out
		if [ -s Bedaux.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Blevins's version            "
	if [ -s Blevins.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Blevins.out
		if [ -s Blevins.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Cokus's version              "
	if [ -s Cokus.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Cokus.out
		if [ -s Cokus.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Kuenning's version           "
	if [ -s Kuenning.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Kuenning.out
		if [ -s Kuenning.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Ladd's version               "
	if [ -s Ladd.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Ladd.out
		if [ -s Ladd.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Yang's version               "
	if [ -s Yang.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Yang.out
		if [ -s Yang.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
	
	printf "  Built-in rand()              "
	if [ -s Standard.out ]; then
		$AWK '/Time elapsed/{ printf( "%6.1f million per second", 1000 / $4 ) }' Standard.out
		if [ -s Standard.reference.tmp ]; then
			printf " (non-conforming)\n"
		else
			printf "\n"
		fi
	else
		printf "       Failed to run\n"
	fi
fi

# Report own performance test results

echo

grep -A 100 "Test of generation rates in various distributions:" Wagner.out

# Summarize results

echo

grep "Error" Wagner.out >> Wagner.error.tmp

if [ -s Wagner.reference.tmp ] && ![ -s Wagner.error.tmp ]; then
	cat Wagner.out Original.out Cokus.out > Bug.txt
	echo "Failed tests - MersenneTwister.h generated incorrect output"
	echo "Results have been written to a file called 'Bug.txt'"
	echo "Please send a copy of 'Bug.txt' and system info to wagnerr@umich.edu"
else
	echo "MersenneTwister.h passed all tests"
fi

rm -f *.tmp

exit 0
