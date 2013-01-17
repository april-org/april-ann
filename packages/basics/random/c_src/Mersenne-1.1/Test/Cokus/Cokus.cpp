// Cokus.cpp
// Test output and speed of Cokus's Mersenne Twister program
// Richard J. Wagner  28 Sep 2009

// This is free, unrestricted software void of any warranty.

#include <cstdio>
#include <ctime>
#include "mt19937ar-cok.c"

int main ( int argc, char * const argv[] )
{
	printf( "Testing output and speed of Cokus's Mersenne Twister program\n" );
	printf( "\nTest of random integer generation:\n" );
	
	unsigned long oneSeed = 4357UL;
	unsigned long bigSeed[4] = { 0x123UL, 0x234UL, 0x345UL, 0x456UL };
	
	printf( "\nTest of random integer generation:\n" );
	init_by_array( bigSeed, 4 );
	unsigned long i;
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10lu ", genrand_int32() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1) generation:\n" );
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10.8f ", genrand_real2() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1] generation:\n" );
	init_genrand( oneSeed );
	for( i = 0; i < 2000UL; ++i )
	{
		printf( "%10.8f ", genrand_real1() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of time to generate one billion random integers:\n" );
	init_genrand( oneSeed );
	unsigned long junk = 0;
	clock_t start = clock();
	for( i = 0; i < 1000000000UL; ++i )
	{
		junk ^= genrand_int32();
	}
	clock_t stop = clock();
	if( junk == 0 ) printf( "jinx\n" );
	printf( "Time elapsed = " );
	printf( "%8.3f", double( stop - start ) / CLOCKS_PER_SEC );
	printf( " s\n" );
	
	return 0;
}
