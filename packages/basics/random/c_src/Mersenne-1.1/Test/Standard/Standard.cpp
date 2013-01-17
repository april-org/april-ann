// Standard.cpp
// Test output and speed of standard random number generator
// Richard J. Wagner  29 Sep 2009

// This is free, unrestricted software void of any warranty.

#include <cstdio>
#include <cstdlib>
#include <ctime>

int main ( int argc, char * const argv[] )
{
	printf( "Testing output and speed of standard random number generator\n" );
	printf( "\nTest of random integer generation:\n" );
	
	unsigned long oneSeed = 4357UL;
	unsigned long bigSeed[4] = { 0x123UL, 0x234UL, 0x345UL, 0x456UL };
	
	// NOTE: The standard random number generator provided with a C++ compiler
	// is unlikely to be the Mersenne Twister.  Therefore it probably lacks the
	// ability to accept a big seed and will probably not match any of the
	// expected output.
	
	printf( "\nTest of random integer generation:\n" );
	srand( bigSeed[0] );
	unsigned long i;
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10i ", rand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1) generation:\n" );
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10.8f ", rand() / ( RAND_MAX + 1.0 ) );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1] generation:\n" );
	srand( oneSeed );
	for( i = 0; i < 2000UL; ++i )
	{
		printf( "%10.8f ", rand() / ( RAND_MAX + 0.0 ) );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of time to generate one billion random integers:\n" );
	srand( oneSeed );
	unsigned long junk = 0;
	clock_t start = clock();
	for( i = 0; i < 1000000000UL; ++i )
	{
		junk ^= rand();
	}
	clock_t stop = clock();
	if( junk == 0 ) printf( "jinx\n" );
	printf( "Time elapsed = " );
	printf( "%8.3f", double( stop - start ) / CLOCKS_PER_SEC );
	printf( " s\n" );
	
	return 0;
}
