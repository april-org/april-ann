// Wagner.cpp
// Test output and speed of MersenneTwister.h
// Richard J. Wagner  30 Sep 2009

// This is free, unrestricted software void of any warranty.

#include <cstdio>
#include <ctime>
#include <fstream>
#include "MersenneTwister.h"

void showrate( clock_t start, clock_t stop, int reps );

int main( int argc, char * const argv[] )
{
	printf( "Testing output and speed of MersenneTwister.h\n" );
	
	MTRand::uint32 oneSeed = 4357UL;
	MTRand::uint32 bigSeed[4] = { 0x123UL, 0x234UL, 0x345UL, 0x456UL };
	
	printf( "\nTest of random integer generation:\n" );
	MTRand mtrand1( bigSeed, 4 );
	unsigned long i;
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10lu ", mtrand1.randInt() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1) generation:\n" );
	for( i = 0; i < 1000UL; ++i )
	{
		printf( "%10.8f ", mtrand1.randExc() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of random real number [0,1] generation:\n" );
	MTRand mtrand2( oneSeed );
	for( i = 0; i < 2000UL; ++i )
	{
		printf( "%10.8f ", mtrand2.rand() );
		if( i % 5 == 4 ) printf("\n");
	}
	
	printf( "\nTest of time to generate one billion random integers:\n" );
	MTRand mtrand3( oneSeed );
	unsigned long junk = 0;
	clock_t start = clock();
	for( i = 0; i < 1000000000UL; ++i )
	{
		junk ^= mtrand3.randInt();
	}
	clock_t stop = clock();
	if( junk == 0 ) printf( "jinx\n" );
	printf( "Time elapsed = " );
	printf( "%8.3f", double( stop - start ) / CLOCKS_PER_SEC );
	printf( " s\n" );
	
	printf( "\nTest of generation rates in various distributions:\n" );
	{
		unsigned long junki = 0;
		double junkf = 0;
		
		printf( "  Integers in [0,2^32-1]      " );
		start = clock();
		for( i = 0; i < 1000000000UL; ++i ) junki ^= mtrand3.randInt();
		stop = clock();
		showrate(start,stop,1000);
		
		printf( "  Integers in [0,100]         " );
		start = clock();
		for( i = 0; i < 500000000UL; ++i ) junki ^= mtrand3.randInt(100);
		stop = clock();
		showrate(start,stop,500);
		
		printf( "  Reals in [0,1]              " );
		start = clock();
		for( i = 0; i < 500000000UL; ++i ) junkf += mtrand3.rand();
		stop = clock();
		showrate(start,stop,500);
		
		printf( "  Reals in [0,7]              " );
		start = clock();
		for( i = 0; i < 500000000UL; ++i ) junkf += mtrand3.rand(7.0);
		stop = clock();
		showrate(start,stop,500);
		
		printf( "  Reals in normal distribution" );
		start = clock();
		for( i = 0; i < 100000000UL; ++i ) junkf += mtrand3.randNorm(7.0,2.0);
		stop = clock();
		showrate(start,stop,100);
		
		if( junki == 0 ) printf("jinx\n");
		if( junkf == 0.0 ) printf("jinx\n");
	}
	
	printf( "\nTests of functionality:\n" );
	
	// Array save/load test
	bool saveArrayFailure = false;
	MTRand mtrand4a, mtrand4b;
	MTRand::uint32 saveArray[ MTRand::SAVE ];
	mtrand4a.save( saveArray );
	mtrand4b.load( saveArray );
	for( i = 0; i < 2000UL; ++i )
		if( mtrand4b.randInt() != mtrand4a.randInt() )
			saveArrayFailure = true;
	if( saveArrayFailure )
		printf( "Error - Failed array save/load test\n" );
	else
		printf( "Passed array save/load test\n" );
	
	// Stream save/load test
	bool saveStreamFailure = false;
	std::ofstream dataOut( "state.dat" );
	if( dataOut )
	{
		dataOut << mtrand4a;  // comment out if compiler does not support
		dataOut.close();
	} 
	std::ifstream dataIn( "state.dat" );
	if( dataIn )
	{
		dataIn >> mtrand4b;  // comment out if compiler does not support
		dataIn.close();
	}
	for( i = 0; i < 2000UL; ++i )
		if( mtrand4b.randInt() != mtrand4a.randInt() )
			saveStreamFailure = true;
	if( saveStreamFailure )
		printf( "Error - Failed stream save/load test\n" );
	else
		printf( "Passed stream save/load test\n" );
	
	// Copy constructor test
	bool copyConstructorFailure = false;
	MTRand mtrand4c( mtrand4a );
	for( i = 0; i < 2000UL; ++i )
		if( mtrand4c.randInt() != mtrand4a.randInt() )
			copyConstructorFailure = true;
	if( copyConstructorFailure )
		printf( "Error - Failed copy constuctor test\n" );
	else
		printf( "Passed copy constructor test\n" );
	
	// Copy operator test
	bool copyOperatorFailure = false;
	MTRand mtrand4d;
	mtrand4d = mtrand4a;
	for( i = 0; i < 2000UL; ++i )
		if( mtrand4d.randInt() != mtrand4a.randInt() )
			copyOperatorFailure = true;
	if( copyOperatorFailure )
		printf( "Error - Failed copy operator test\n" );
	else
		printf( "Passed copy operator test\n" );
	
	// Integer range test
	MTRand mtrand5;
	bool integerRangeFailure = false;
	unsigned long integerRangeCount[18];
	for( i = 0; i < 18UL; ++i )
		integerRangeCount[i] = 0;
	for( i = 0; i < 180000UL; ++i )
	{
		int r = mtrand5.randInt(17);
		if( r < 0 || r > 17 )
			integerRangeFailure = true;
		else
			++integerRangeCount[ mtrand5.randInt(17) ];
	}
	for( i = 0; i < 18UL; ++i )
		if( integerRangeCount[i] < 5000 || integerRangeCount[i] > 15000 )
			integerRangeFailure = true;
	if( integerRangeFailure )
		printf( "Error - Failed integer range test\n" );
	else
		printf( "Passed integer range test\n" );
	
	// Float range test
	MTRand mtrand6;
	bool floatRangeFailure = false;
	unsigned long floatRangeLo = 0;
	unsigned long floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6();
		if( r < 0.0 || r > 1.0 )
			floatRangeFailure = true;
		if( r < 0.1 ) ++floatRangeLo;
		if( r > 0.9 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	floatRangeLo = 0;
	floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.rand();
		if( r < 0.0 || r > 1.0 )
			floatRangeFailure = true;
		if( r < 0.1 ) ++floatRangeLo;
		if( r > 0.9 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	floatRangeLo = 0;
	floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.rand(0.3183);
		if( r < 0.0 || r > 0.3183 )
			floatRangeFailure = true;
		if( r < 0.03183 ) ++floatRangeLo;
		if( r > 0.28647 ) ++floatRangeHi;
	}
	floatRangeFailure |= !floatRangeLo;
	floatRangeFailure |= !floatRangeHi;
	floatRangeLo = false;
	floatRangeHi = false;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.randExc();
		if( r < 0.0 || r >= 1.0 )
			floatRangeFailure = true;
		if( r < 0.1 ) ++floatRangeLo;
		if( r > 0.9 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	floatRangeLo = 0;
	floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.randExc(0.3183);
		if( r < 0.0 || r >= 0.3183 )
			floatRangeFailure = true;
		if( r < 0.03183 ) ++floatRangeLo;
		if( r > 0.28647 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	floatRangeLo = 0;
	floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.randDblExc();
		if( r <= 0.0 || r >= 1.0 )
			floatRangeFailure = true;
		if( r < 0.1 ) ++floatRangeLo;
		if( r > 0.9 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	floatRangeLo = 0;
	floatRangeHi = 0;
	for( i = 0; i < 100000UL; ++i )
	{
		double r = mtrand6.randDblExc(0.3183);
		if( r <= 0.0 || r >= 0.3183 )
			floatRangeFailure = true;
		if( r < 0.03183 ) ++floatRangeLo;
		if( r > 0.28647 ) ++floatRangeHi;
	}
	if( floatRangeLo < 5000 || floatRangeLo > 15000 )
		floatRangeFailure = true;
	if( floatRangeHi < 5000 || floatRangeHi > 15000 )
		floatRangeFailure = true;
	if( floatRangeFailure )
		printf( "Error - Failed float range test\n" );
	else
		printf( "Passed float range test\n" );
	
	// Gaussian distribution test
	MTRand mtrand7;
	bool gaussianFailure = false;
	const double mean = 100.0;
	const double stddev = 50.0;
	long gaussianExpect[] =
		{ 0, 0, 0, 0, 0, 0, 0, 31, 1318, 21400, 135905, 341344,
		  341344, 135905, 21400, 1318, 31, 0, 0, 0, 0, 0, 0, 0 };
	long gaussian[24];
	for( i = 0; i < 24; ++i )
		gaussian[i] = 0;
	for( i = 0; i < 1000000UL; ++i )
	{
		double r = mtrand7.randNorm( mean, stddev );
		long bin = 12 + long( floor( ( r - mean ) / stddev ) );
		if( bin < 0 || bin > 23 )
			gaussianFailure = true;
		else
			++gaussian[bin];
	}
	for( i = 0; i < 24UL; ++i )
	{
		if( gaussian[i] - 2 * gaussianExpect[i] > 5 )
			gaussianFailure = true;
		if( gaussianExpect[i] / 2 - gaussian[i] > 5 )
			gaussianFailure = true;
	}
	if( gaussianFailure )
		printf( "Error - Failed Gaussian distribution test\n" );
	else
		printf( "Passed Gaussian distribution test\n" );
	
	// Auto-seed uniqueness test
	MTRand mtrand8a, mtrand8b, mtrand8c;
	unsigned long r8a = mtrand8a.randInt();
	unsigned long r8b = mtrand8b.randInt();
	unsigned long r8c = mtrand8c.randInt();
	if( r8a == r8b || r8a == r8c || r8b == r8c )
		printf( "Error - Failed auto-seed uniqueness test\n" );
	else
		printf( "Passed auto-seed uniqueness test\n" );
	
	return 0;
}

void showrate( clock_t start, clock_t stop, int reps )
{
	double time = double( stop - start ) / CLOCKS_PER_SEC;
	double rate = reps / time;
	printf( " %6.1f million per second\n", rate );
}
