// Kuenning.cpp
// Test output and speed of Kuenning's Mersenne Twister class
// Richard J. Wagner  28 Sep 2009

// This is free, unrestricted software void of any warranty.

#include <cstdio>
#include <ctime>
#include "mtwist.h"

int main ( int argc, char * const argv[] )
{
	printf( "Testing output and speed of Kuenning's Mersenne Twister class\n" );
	printf( "\nTest of random integer generation:\n" );
	
	unsigned long oneSeed = 4357UL;
	mt_u32bit_t bigSeed[MT_STATE_SIZE] = { 0x123, 0x234, 0x345, 0x456 };
	
	// NOTE: Kuenning's class lacks the usual seeding and generation functions
	
	printf( "\nTest of random integer generation:\n" );
	mt_prng mtrand1( bigSeed );
	unsigned long i;
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10lu ", mtrand1.lrand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1) generation:\n" );
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10.8f ", mtrand1.drand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1] generation:\n" );
	mt_prng mtrand2( oneSeed );
	for( i = 0; i < 2000UL; ++i )
	{
		printf( "%10.8f ", mtrand2.drand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of time to generate one billion random integers:\n" );
	mt_prng mtrand3( oneSeed );
	unsigned long junk = 0;
	clock_t start = clock();
	for( i = 0; i < 1000000000UL; ++i )
	{
		junk ^= mtrand3.lrand();
	}
	clock_t stop = clock();
	if( junk == 0 ) printf( "jinx\n" );
	printf( "Time elapsed = " );
	printf( "%8.3f", double( stop - start ) / CLOCKS_PER_SEC );
	printf( " s\n" );
	
	return 0;
}
