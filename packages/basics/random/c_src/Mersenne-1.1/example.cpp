// example.cpp
// Examples of random number generation with MersenneTwister.h
// Richard J. Wagner  27 September 2009

#include <iostream>
#include <fstream>
#include "MersenneTwister.h"

using namespace std;

int main( int argc, char * const argv[] )
{
	// A Mersenne Twister random number generator
	// can be declared with a simple
	
	MTRand mtrand1;
	
	// and used with
	
	double a = mtrand1();
	
	// or
	
	double b = mtrand1.rand();
	
	cout << "Two real numbers in the range [0,1]:  ";
	cout << a << ", " << b << endl;
	
	// Those calls produced the default of floating-point numbers
	// in the range 0 to 1, inclusive.  We can also get integers
	// in the range 0 to 2^32 - 1 (4294967295) with
	
	unsigned long c = mtrand1.randInt();
	
	cout << "An integer in the range [0," << 0xffffffffUL;
	cout << "]:  " << c << endl;
	
	// Or get an integer in the range 0 to n (for n < 2^32) with
	
	int d = mtrand1.randInt( 42 );
	
	cout << "An integer in the range [0,42]:  " << d << endl;
	
	// We can get a real number in the range 0 to 1, excluding
	// 1, with
	
	double e = mtrand1.randExc();
	
	cout << "A real number in the range [0,1):  " << e << endl;
	
	// We can get a real number in the range 0 to 1, excluding
	// both 0 and 1, with
	
	double f = mtrand1.randDblExc();
	
	cout << "A real number in the range (0,1):  " << f << endl;
	
	// The functions rand(), randExc(), and randDblExc() can
	// also have ranges defined just like randInt()
	
	double g = mtrand1.rand( 2.5 );
	double h = mtrand1.randExc( 10.0 );
	double i = 12.0 + mtrand1.randDblExc( 8.0 );
	
	cout << "A real number in the range [0,2.5]:  " << g << endl;
	cout << "One in the range [0,10.0):  " << h << endl;
	cout << "And one in the range (12.0,20.0):  " << i << endl;
	
	// The distribution of numbers over each range is uniform,
	// but it can be transformed to other useful distributions.
	// One common transformation is included for drawing numbers
	// in a normal (Gaussian) distribution
	
	cout << "A few grades from a class with a 52 pt average ";
	cout << "and a 9 pt standard deviation:" << endl;
	for( int student = 0; student < 20; ++student )
	{
		double j = mtrand1.randNorm( 52.0, 9.0 );
		cout << ' ' << int(j);
	}
	cout << endl;
	
	// Random number generators need a seed value to start
	// producing a sequence of random numbers.  We gave no seed
	// in our declaration of mtrand1, so one was automatically
	// generated from the system clock (or the operating system's
	// random number pool if available).  Alternatively we could
	// provide our own seed.  Each seed uniquely determines the
	// sequence of numbers that will be produced.  We can
	// replicate a sequence by starting another generator with
	// the same seed.
	
	MTRand mtrand2a( 1973 );  // makes new MTRand with given seed
	
	double k1 = mtrand2a();   // gets the first number generated
	
	MTRand mtrand2b( 1973 );  // makes an identical MTRand
	
	double k2 = mtrand2b();   // and gets the same number
	
	cout << "These two numbers are the same:  ";
	cout << k1 << ", " << k2 << endl;
	
	// We can also restart an existing MTRand with a new seed
	
	mtrand2a.seed( 1776 );
	mtrand2b.seed( 1941 );
	
	double l1 = mtrand2a();
	double l2 = mtrand2b();
	
	cout << "Re-seeding gives different numbers:  ";
	cout << l1 << ", " << l2 << endl;
	
	// But there are only 2^32 possible seeds when we pass a
	// single 32-bit integer.  Since the seed dictates the
	// sequence, only 2^32 different random number sequences will
	// result.  For applications like Monte Carlo simulation we
	// might want many more.  We can seed with an array of values
	// rather than a single integer to access the full 2^19937-1
	// possible sequences.
	
	MTRand::uint32 seed[ MTRand::N ];
	for( int n = 0; n < MTRand::N; ++n )
		seed[n] = 23 * n;  // fill with anything
	MTRand mtrand3( seed );
	
	double m1 = mtrand3();
	double m2 = mtrand3();
	double m3 = mtrand3();
	
	cout << "We seeded this sequence with 19968 bits:  ";
	cout << m1 << ", " << m2 << ", " << m3 << endl;
	
	// Again we will have the same sequence every time we run the
	// program.  Make the array with something that will change
	// to get unique sequences.  On a Linux system, the default
	// auto-initialization routine takes a unique sequence from
	// /dev/urandom.
	
	// For cryptography, also remember to hash the generated
	// random numbers.  Otherwise the internal state of the
	// generator can be learned after reading 624 values.
	
	// We might want to save the state of the generator at an
	// arbitrary point after seeding so a sequence could be
	// replicated.  An MTRand object can be saved into an array
	// or to a stream.  
	
	MTRand mtrand4;
	
	// The array must be of type uint32 and length SAVE.
	
	MTRand::uint32 randState[ MTRand::SAVE ];
	
	mtrand4.save( randState );
	
	// A stream is convenient for saving to a file.
	
	ofstream stateOut( "state.dat" );
	if( stateOut )
	{
		stateOut << mtrand4;
		stateOut.close();
	}
	
	unsigned long n1 = mtrand4.randInt();
	unsigned long n2 = mtrand4.randInt();
	unsigned long n3 = mtrand4.randInt();
	
	cout << "A random sequence:       "
	     << n1 << ", " << n2 << ", " << n3 << endl;
	
	// And loading the saved state is as simple as
	
	mtrand4.load( randState );
	
	unsigned long o4 = mtrand4.randInt();
	unsigned long o5 = mtrand4.randInt();
	unsigned long o6 = mtrand4.randInt();
	
	cout << "Restored from an array:  "
	     << o4 << ", " << o5 << ", " << o6 << endl;
	
	ifstream stateIn( "state.dat" );
	if( stateIn )
	{
		stateIn >> mtrand4;
		stateIn.close();
	}
	
	unsigned long p7 = mtrand4.randInt();
	unsigned long p8 = mtrand4.randInt();
	unsigned long p9 = mtrand4.randInt();
	
	cout << "Restored from a stream:  "
	     << p7 << ", " << p8 << ", " << p9 << endl;
	
	// We can also duplicate a generator by copying
	
	MTRand mtrand5( mtrand3 );  // copy upon construction
	
	double q1 = mtrand3();
	double q2 = mtrand5();
	
	cout << "These two numbers are the same:  ";
	cout << q1 << ", " << q2 << endl;
	
	mtrand5 = mtrand4;  // copy by assignment
	
	double r1 = mtrand4();
	double r2 = mtrand5();
	
	cout << "These two numbers are the same:  ";
	cout << r1 << ", " << r2 << endl;
	
	// In summary, the recommended common usage is
	
	MTRand mtrand6;  // automatically generate seed
	double s = mtrand6();               // real number in [0,1]
	double t = mtrand6.randExc(0.5);    // real number in [0,0.5)
	unsigned long u = mtrand6.randInt(10);  // integer in [0,10]
	
	// with the << and >> operators used for saving to and
	// loading from streams if needed.
	
	cout << "Your lucky number for today is "
	     << s + t * u << endl;
	
	return 0;
}
