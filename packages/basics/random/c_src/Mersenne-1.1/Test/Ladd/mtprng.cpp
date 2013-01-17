//---------------------------------------------------------------------
//  Algorithmic Conjurings @ http://www.coyotegulch.com
//
//  mtprng.cpp (libcoyote)
//
//  Mersenne Twister -- A pseudorandom Number Generator
//
//  ORIGINAL ALGORITHM COPYRIGHT
//  ============================
//  Copyright (C) 1997, 2002 Makoto Matsumoto and Takuji Nishimura.
//  Any feedback is very welcome. For any question, comments, see
//  http://www.math.keio.ac.jp/matumoto/emt.html or email
//  matumoto@math.keio.ac.jp
//---------------------------------------------------------------------
//
//  COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:
//
//  This notice applies *only* to this specific expression of this
//  algorithm, and does not imply ownership or invention of the
//  implemented algorithm.
//  
//  If you modify this file, you may insert additional notices
//  immediately following this sentence.
//  
//  Copyright 2001, 2002, 2003 Scott Robert Ladd.
//  All rights reserved, except as noted herein.
//
//  This computer program source file is supplied "AS IS". Scott Robert
//  Ladd (hereinafter referred to as "Author") disclaims all warranties,
//  expressed or implied, including, without limitation, the warranties
//  of merchantability and of fitness for any purpose. The Author
//  assumes no liability for direct, indirect, incidental, special,
//  exemplary, or consequential damages, which may result from the use
//  of this software, even if advised of the possibility of such damage.
//  
//  The Author hereby grants anyone permission to use, copy, modify, and
//  distribute this source code, or portions hereof, for any purpose,
//  without fee, subject to the following restrictions:
//  
//      1. The origin of this source code must not be misrepresented.
//  
//      2. Altered versions must be plainly marked as such and must not
//         be misrepresented as being the original source.
//  
//      3. This Copyright notice may not be removed or altered from any
//         source or altered source distribution.
//  
//  The Author specifically permits (without fee) and encourages the use
//  of this source code for entertainment, education, or decoration. If
//  you use this source code in a product, acknowledgment is not required
//  but would be appreciated.
//  
//  Acknowledgement:
//      This license is based on the wonderful simple license that
//      accompanies libpng.
//
//-----------------------------------------------------------------------
//
//  For more information on this software package, please visit
//  Scott's web site, Coyote Gulch Productions, at:
//
//      http://www.coyotegulch.com
//  
//-----------------------------------------------------------------------

#include "mtprng.h"
using namespace libcoyote;

#if _MSC_VER < 1300
const mtprng::int_type mtprng::MATRIX_A    = 0x9908b0dfUL;
const mtprng::int_type mtprng::UPPER_MASK  = 0x80000000UL;
const mtprng::int_type mtprng::LOWER_MASK  = 0x7fffffffUL;
#endif

//--------------------------------------------------------------------------
//  Constructor
mtprng::mtprng(int_type seed)
{
    init(seed);
}

//--------------------------------------------------------------------------
//  Initializes the generator with "seed"
void mtprng::init(int_type seed)
{
    // Save seed for historical purpose
    m_seed = seed;
    m_mt[0] = seed & 0xffffffffUL;
    
    // Set the seed using values suggested by Matsumoto & Nishimura, using
    //   a generator by Knuth. See original source for details.
    for (m_mti = 1; m_mti < N; ++m_mti)
        m_mt[m_mti] = 0xffffffffUL & (1812433253UL * (m_mt[m_mti - 1] ^ (m_mt[m_mti - 1] >> 30)) + m_mti);
        
}

//--------------------------------------------------------------------------
// Initialize the generator from an array of seeds
void mtprng::init_by_array(const std::vector<int_type> & init_key)
{
    // Note: variable names match those in original example
    size_t i = 1;
    size_t j = 0;
    size_t k = static_cast<size_t>((N > init_key.size()) ? N : init_key.size());
    init(19650218UL);
    
    for (; k; --k)
    {
        m_mt[i] = (m_mt[i] ^ ((m_mt[i - 1] ^ (m_mt[i - 1] >> 30)) * 1664525UL)) + init_key[j] + j;
        m_mt[i] &= 0xffffffffUL;
        ++i;
        ++j;
        
        if (i >= N)
        {
            m_mt[0] = m_mt[N - 1];
            i = 1;
        }
        
        if (j >= static_cast<size_t>(init_key.size()))
            j = 0;
    }
    
    for (k = N - 1; k; --k)
    {
        m_mt[i] = (m_mt[i] ^ ((m_mt[i - 1] ^ (m_mt[i - 1] >> 30)) * 1566083941UL)) - i;
        m_mt[i] &= 0xffffffffUL;
        ++i;
        
        if (i >= N)
        {
            m_mt[0] = m_mt[N - 1];
            i = 1;
        }
    }
    
    m_mt[0] = 0x80000000UL;
}
    
//--------------------------------------------------------------------------
//   Obtain the next 32-bit integer in the psuedo-random sequence
mtprng::int_type mtprng::get_rand()
{
    // Note: variable names match those in original example
    const int_type mag01[2] = { 0, MATRIX_A };
    int_type y;
    size_t   kk;
    
    // Generate N words at a time
    if (m_mti >= N)
    {
        // Fill the m_mt array
        for (kk=0; kk < N-M; kk++)
        {
            y = (m_mt[kk] & UPPER_MASK) | (m_mt[kk+1] & LOWER_MASK);
            m_mt[kk] = m_mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1];
        }

        for ( ; kk < N-1; kk++)
        {
            y = (m_mt[kk] & UPPER_MASK) | (m_mt[kk+1] & LOWER_MASK);
            m_mt[kk] = m_mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1];
        }

        y = (m_mt[N-1] & UPPER_MASK) | (m_mt[0]&LOWER_MASK);
        m_mt[N-1] = m_mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1];

        m_mti = 0;
    }
    
    // Here is where we actually calculate the number with a series of transformations 
    y = m_mt[m_mti++];

    y ^= (y >> 11);
    y ^= (y <<  7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y; 
}

//--------------------------------------------------------------------------
//   Obtain a psuedorandom real number in the range [0,1), i.e., a number
//   greater than or equal to 0 and less than 1, with 53-bit precision.
mtprng::real_type mtprng::get_rand_real53()
{
    // privides a granularity of approx. 1.1E-16
    int_type a = get_rand() >> 5;
    int_type b = get_rand() >> 6;
    return real_type(a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

