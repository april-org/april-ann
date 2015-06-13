#include "buffer_list.h"
#include "MersenneTwister.h"

namespace Basics {

  const unsigned int MTRand::N;
  const unsigned int MTRand::SAVE;

  MTRand::MTRand(const uint32_t& oneSeed) {
    seed(oneSeed); 
  }

  MTRand::MTRand(uint32_t *const bigSeed, const uint32_t seedLength) {
    seed(bigSeed,seedLength);
  }

  MTRand::MTRand() { 
    seed(); 
  }

  MTRand::MTRand(const MTRand& r) : Serializable() {
    register uint32_t *s = state;
    const register uint32_t *t = r.state;
    register int i = N;
    for( ; i--; *s++ = *t++ ) {}
    left = r.left;
    pNext = &state[N-left];
  }

  MTRand& MTRand::operator=(const MTRand& r) {
    if (this != &r) {
      register uint32_t *s = state;
      const register uint32_t *t = r.state;
      register int i = N;
      for( ; i--; *s++ = *t++ ) {}
      left = r.left;
      pNext = &state[N-left];
    }
    return (*this);
  }

  double MTRand::rand() {
    return double(randInt()) * (1.0/4294967295.0); 
  }

  double MTRand::rand( const double& n) {
    return rand() * n;
  }

  double MTRand::randExc() {
    return double(randInt()) * (1.0/4294967296.0);
  }

  double MTRand::randExc(const double& n) {
    return randExc() * n; 
  }

  double MTRand::randDblExc() {
    return ( double(randInt()) + 0.5 ) * (1.0/4294967296.0); 
  }

  double MTRand::randDblExc(const double& n) {
    return randDblExc() * n; 
  }

  double MTRand::rand53() {
    uint32_t a = randInt() >> 5, b = randInt() >> 6;
    return ( a * 67108864.0 + b ) * (1.0/9007199254740992.0);  // by Isaku Wada
  }

  double MTRand::randNorm(const double& mean, const double& variance) {
    // Return a real number from a normal (Gaussian) distribution with given
    // mean and variance by Box-Muller method
    double r = sqrt( -2.0 * log( 1.0-randDblExc()) ) * variance;
    double phi = 2.0 * 3.14159265358979323846264338328 * randExc();
    return mean + r * cos(phi);
  }

  uint32_t MTRand::randInt() {
    // Pull a 32-bit integer from the generator state
    // Every other access function simply transforms the numbers extracted here
	
    if( left == 0 ) reload();
    --left;
		
    register uint32_t s1;
    s1 = *pNext++;
    s1 ^= (s1 >> 11);
    s1 ^= (s1 <<  7) & 0x9d2c5680UL;
    s1 ^= (s1 << 15) & 0xefc60000UL;
    return ( s1 ^ (s1 >> 18) );
  }

  uint32_t MTRand::randInt(const uint32_t& n) {
    // Find which bits are used in n
    uint32_t used = n;
    used |= used >> 1;
    used |= used >> 2;
    used |= used >> 4;
    used |= used >> 8;
    used |= used >> 16;
	
    // Draw numbers until one is found in [0,n]
    uint32_t i;
    do
      i = randInt() & used;  // toss unused bits to shorten search
    while( i > n );
    return i;
  }


  void MTRand::seed( const uint32_t oneSeed ) {
    // Seed the generator with a simple uint32
    initialize(oneSeed);
    reload();
  }


  void MTRand::seed( uint32_t *const bigSeed, const uint32_t seedLength ) {
    // Seed the generator with an array of uint32's
    // There are 2^19937-1 possible initial states.  This function allows
    // all of those to be accessed by providing at least 19937 bits (with a
    // default seed length of N = 624 uint32's).  Any bits above the lower 32
    // in each element are discarded.
    // Just call seed() if you want to get array from /dev/urandom
    initialize(19650218UL);
    register unsigned int i = 1;
    register uint32_t j = 0;
    register int k = ( N > seedLength ? N : seedLength );
    for( ; k; --k )
      {
        state[i] =
          state[i] ^ ( (state[i-1] ^ (state[i-1] >> 30)) * 1664525UL );
        state[i] += ( bigSeed[j] & 0xffffffffUL ) + j;
        state[i] &= 0xffffffffUL;
        ++i;  ++j;
        if( i >= N ) { state[0] = state[N-1];  i = 1; }
        if( j >= seedLength ) j = 0;
      }
    for( k = N - 1; k; --k )
      {
        state[i] =
          state[i] ^ ( (state[i-1] ^ (state[i-1] >> 30)) * 1566083941UL );
        state[i] -= i;
        state[i] &= 0xffffffffUL;
        ++i;
        if( i >= N ) { state[0] = state[N-1];  i = 1; }
      }
    state[0] = 0x80000000UL;  // MSB is 1, assuring non-zero initial array
    reload();
  }


  void MTRand::seed() {
    // Seed the generator with an array from /dev/urandom if available
    // Otherwise use a hash of time() and clock() values
	
    // First try getting an array from /dev/urandom
    FILE* urandom = fopen( "/dev/urandom", "rb" );
    if( urandom )
      {
        uint32_t bigSeed[N];
        register uint32_t *s = bigSeed;
        register int i = N;
        register bool success = true;
        while( success && i-- )
          success = fread( s++, sizeof(uint32_t), 1, urandom );
        fclose(urandom);
        if( success ) { seed( bigSeed, N );  return; }
      }
	
    // Was not successful, so use time() and clock() instead
    seed( hash( time(NULL), clock() ) );
  }


  void MTRand::initialize( const uint32_t seed ) {
    // Initialize generator state with seed
    // See Knuth TAOCP Vol 2, 3rd Ed, p.106 for multiplier.
    // In previous versions, most significant bits (MSBs) of the seed affect
    // only MSBs of the state array.  Modified 9 Jan 2002 by Makoto Matsumoto.
    register uint32_t *s = state;
    register uint32_t *r = state;
    register unsigned int i = 1;
    *s++ = seed & 0xffffffffUL;
    for( ; i < N; ++i )
      {
        *s++ = ( 1812433253UL * ( *r ^ (*r >> 30) ) + i ) & 0xffffffffUL;
        r++;
      }
  }


  void MTRand::reload() {
    // Generate N new values in state
    // Made clearer and faster by Matthew Bellew (matthew.bellew@home.com)
    static const int MmN = int(M) - int(N);  // in case enums are unsigned
    register uint32_t *p = state;
    register int i;
    for( i = N - M; i--; ++p )
      *p = twist( p[M], p[0], p[1] );
    for( i = M; --i; ++p )
      *p = twist( p[MmN], p[0], p[1] );
    *p = twist( p[MmN], p[0], state[0] );

    left = N, pNext = state;
  }


  uint32_t MTRand::hash( time_t t, clock_t c ) {
    // Get a uint32 from t and c
    // Better than uint32(x) in case x is floating point in [0,1]
    // Based on code by Lawrence Kirby (fred@genesis.demon.co.uk)

    static uint32_t differ = 0;  // guarantee time-based seeds will change

    uint32_t h1 = 0;
    unsigned char *p = (unsigned char *) &t;
    for( size_t i = 0; i < sizeof(t); ++i )
      {
        h1 *= UCHAR_MAX + 2U;
        h1 += p[i];
      }
    uint32_t h2 = 0;
    p = (unsigned char *) &c;
    for( size_t j = 0; j < sizeof(c); ++j ) {
      h2 *= UCHAR_MAX + 2U;
      h2 += p[j];
    }
    return ( h1 + differ++ ) ^ h2;
  }

  void MTRand::shuffle(int size, int *vector) {
    for (int i=0; i < size; i++)
      vector[i] = i;
    for (int i = size-1; i > 0; i--) {
      int j = randInt(i);
      int swap = vector[i];
      vector[i] = vector[j];
      vector[j] = swap;
    }
  }

  void MTRand::save( uint32_t* saveArray ) const {
    register uint32_t *sa = saveArray;
    register const uint32_t *s = state;
    register int i = N;
    for( ; i--; *sa++ = *s++ ) {}
    *sa = left;
  }

  void MTRand::load( uint32_t *const loadArray ) {
    register uint32_t *s = state;
    register uint32_t *la = loadArray;
    register int i = N;
    for( ; i--; *s++ = *la++ ) {}
    left = *la;
    pNext = &state[N-left];
  }

  const char *MTRand::luaCtorName() const {
    return "random():fromTable";
  }
  
  int MTRand::exportParamsToLua(lua_State *L) {
    uint32_t randState[ SAVE ];
    save( randState );
    AprilUtils::LuaTable table(L);
    for (unsigned int i=0; i<SAVE; ++i) {
      table[i+1] = randState[i];
    }
    table.pushTable(L);
    return 1;
  }

} // namespace Basics
