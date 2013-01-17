// Ladd.cpp
// Test output and speed of Ladd's Mersenne Twister class
// Richard J. Wagner  28 Sep 2009

// This is free, unrestricted software void of any warranty.

#include <cstdio>
#include <ctime>
#include <vector>
#include "mtprng.h"

using namespace libcoyote;
using namespace std;

int main ( int argc, char * const argv[] )
{
	printf( "Testing output and speed of Ladd's Mersenne Twister class\n" );
	printf( "\nTest of random integer generation:\n" );
	
	unsigned long oneSeed = 4357UL;
	vector<unsigned long> bigSeed;
	bigSeed.push_back( 0x123UL );
	bigSeed.push_back( 0x234UL );
	bigSeed.push_back( 0x345UL );
	bigSeed.push_back( 0x456UL );
	
	printf( "\nTest of random integer generation:\n" );
	mtprng mtrand1;
	mtrand1.init_by_array( bigSeed );
	unsigned long i;
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10lu ", mtrand1.get_rand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1) generation:\n" );
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10.8f ", mtrand1.get_rand_real2() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1] generation:\n" );
	mtprng mtrand2( oneSeed );
	for( i = 0; i < 2000UL; ++i )
	{
		printf( "%10.8f ", mtrand2.get_rand_real1() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of time to generate one billion random integers:\n" );
	mtprng mtrand3( oneSeed );
	unsigned long junk = 0;
	clock_t start = clock();
	for( i = 0; i < 1000000000UL; ++i )
	{
		junk ^= mtrand3.get_rand();
	}
	clock_t stop = clock();
	if( junk == 0 ) printf( "jinx\n" );
	printf( "Time elapsed = " );
	printf( "%8.3f", double( stop - start ) / CLOCKS_PER_SEC );
	printf( " s\n" );
	
	return 0;
}
