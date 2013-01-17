//---------------------------------------------------------------------
//  Algorithmic Conjurings @ http://www.coyotegulch.com
//
//  mtprng.h (libcoyote)
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
//  For more information on this software package, please visit
//  Scott's web site, Coyote Gulch Productions, at:
//
//      http://www.coyotegulch.com
//-----------------------------------------------------------------------

#if !defined(LIBCOYOTE_MTPRNG_H)
#define LIBCOYOTE_MTPRNG_H

// Standard C++ Library
#include <cmath>
#include <vector>
#include <cstddef>

namespace libcoyote
{
    //! Implements the Mersenne Twister, a peudorandom number generator
    /*!
        The mtprng class encapsulates the Mersenne Twister algorithm invented by
        Makoto Matsumoto and Takuji Nishimura. One of the appealing aspects of the
        Mersenne Twister is its use of binary operations (as opposed to
        time-consuming multiplication) for generating numbers. The algorithm's
        period is 2<sup>19937</sup>-1 (~10<sup>6001</sup>), as compared to a
        period of ~10<sup>8</sup> for the best variants of the linear congruential
        methods. For practical purposes, the Mersenne Twister doesn't repeat 
        itself. And the basic algorithm generates 32-bit integers, while the
        standard rand() function produces 16-bit values.
        <p>
        <b><i>Version history</i></b>
        - 1.0.0 (9 December 2001)
            - Original release
            .
        - 1.1.0 (26 Jan 2002)
            - Fixed problems with Microsoft C++ 6 (v12.00) and static constant initializations.
            - Added protection from multiple inclusions.
            - Added Doxygen comments.
            - Changed get_rand_range to use int_value parameters instead of int.
            - Added function to return an encoded version number.
            .
        1.2.0 (5 February 2002)
            - Implemented enhancements announced on 26 Jan by Matsumoto and Nishimura
            - Added new functions to reflect M&N's latest work.
            .
        1.2.1   7 February 2002
            - Added function to return a random zero-based, size_t index
            - Inlined the simplest "get" functions
            .
        1.3.0   19 April 2002
            - Updated for improve Microsoft Visual C++ 13.0 in Visual Studio.NET.
            - Move these classes back into the libcoyote namespace.
            .
        1.3.1   15 June 2002
            - Updated and expanded doxygen comments
            .
        1.3.2   15 June 2002
            - Added ability to retrieve seed used in initialization
            .
        1.3.2   15 January 2003
            - Modified init_by_array to eliminate warnings.
            - improved consistency with my other code.
            .
        \version 1.3.2
        \date    15 January 2003
    */
    class mtprng
    {
    public:
        //! Defines the 32-bit integer type.
        typedef unsigned long int_type;  // needs to be a 32-bit type

        //! Defines the 64-bit IEEE-754 type.
        typedef double        real_type; // needs to be a 64-bit or larger type

    private:
        #if _MSC_VER < 1300
        // Period parameters
        #define N size_t(624)
        #define M size_t(397)

        static const int_type MATRIX_A;
        static const int_type UPPER_MASK;
        static const int_type LOWER_MASK;
        #else
        // Period parameters
        static const size_t N = 624;
        static const size_t M = 397;

        static const int_type MATRIX_A   = 0x9908b0dfUL;
        static const int_type UPPER_MASK = 0x80000000UL;
        static const int_type LOWER_MASK = 0x7fffffffUL;
        #endif

        // Working storage
        int_type m_seed;
        int_type m_mt[N];
        size_t   m_mti;
        int_type m_multiplier;

    public:
        //! Default constructor, with optional seed.
        /*!
            The constructor uses a default value for the seed. In practice,
            you'll want to use a better seed, perhaps based on the time or
            some stochastic source such as /dev/random or /dev/urandom.
            \param seed - Seed value used to "start" or seed the generator
        */
        mtprng(int_type seed = 19650218UL);

        //! Initializes the generator with "seed"
        /*!
            Resets the generator using the provided seed value.
            \param seed - Seed value used to "start" or seed the generator
            \sa int_type
        */
        void init(int_type seed);

        //! Returns the original seed value
        /*!
            Returns the seed value used to initialize this generator.
            \return The seed value used to initialize this generator
        */
        int_type get_seed()
        {
            return m_seed;
        }
        
        //! Initialize the generator from an array of seeds
        /*!
            Uses an array of integer seeds to initialize the generator.
            \param init_key - A vector of int_type seeds
            \sa int_type
        */
        void init_by_array(const std::vector<int_type> & init_key);
    
        //!  Get the next integer
        /*!
            Returns the next int_type in sequence.
            \return A pseudorandom int_type value
            \sa int_type
        */
        int_type get_rand();

        //! Get the next integer in the range [lo,hi]
        /*!
            Returns the next int_value between lo and hi, inclusive.
            \param lo - Minimum value of result
            \param hi - Maximum value of result
            \return A pseudorandom int_type value
            \sa int_type
        */
        int_type get_rand_range(int_type lo, int_type hi);

        //! Get the next random value as a size_t index
        /*!
            Returns the next value as a size_t "index" in the range [0,length).
            \param length - Maximum value of result
            \return A pseudorandom size_t value
        */
        size_t get_rand_index(size_t length);

        //! Get the next number in the range [0,1]
        /*!
            Returns the next real number in the range [0,1], i.e., a number
            greater than or equal to 0 and less than or equal to 1.
            Provides 32-bit precision.
            \return A pseudorandom real_type value
            \sa real_type
        */
        real_type get_rand_real1();
    
        //! Get the next number in the range [0,1)
        /*!
            Returns the next real number in the range [0,1), i.e., a number
            greater than or equal to 0 and less than 1.
            Provides 32-bit precision.
            \return A pseudorandom real_type value
            \sa real_type
        */
        real_type get_rand_real2();
    
        //! Get the next number in the range (0,1)
        /*!
            Returns the next real number in the range (0,1), i.e., a number
            greater than 0 and less than 1.
            Provides 32-bit precision.
            \return A pseudorandom real_type value
            \sa real_type
        */
        real_type get_rand_real3();
    
        //! Get the next number in the range [0,1)
        /*!
            Returns the next real number in the range [0,1), i.e., a number
            greater than or equal to 0 and less than 1.
            Provides 53-bit precision.
            \return A pseudorandom real_type value
            \sa real_type
        */
        real_type get_rand_real53();
    };

    //---------------------------------------------------------------------------
    //   Obtain a psuedorandom integer in the range [lo,hi]
    inline mtprng::int_type mtprng::get_rand_range(mtprng::int_type lo, mtprng::int_type hi)
    {
        // Local working storage
        real_type range = hi - lo + 1.0;
    
        // Use real value to caluclate range
        return lo + int_type(floor(range * get_rand_real2()));
    }

    //--------------------------------------------------------------------------
    //  Returns the next value as a size_t "index" in the range [0,length).
    inline size_t mtprng::get_rand_index(size_t length)
    {
        return size_t(real_type(length) * get_rand_real2());
    }

    //--------------------------------------------------------------------------
    //   Obtain a psuedorandom real number in the range [0,1], i.e., a number
    //   greater than or equal to 0 and less than or equal to 1.
    inline mtprng::real_type mtprng::get_rand_real1()
    {
        // privides a granularity of approx. 2.3E-10
        return real_type(get_rand()) * (1.0 / 4294967295.0);
    }

    //--------------------------------------------------------------------------
    //   Obtain a psuedorandom real number in the range [0,1), i.e., a number
    //   greater than or equal to 0 and less than 1.
    inline mtprng::real_type mtprng::get_rand_real2()
    {
        // privides a granularity of approx. 2.3E-10
        return real_type(get_rand()) * (1.0 / 4294967296.0);
    }

    //--------------------------------------------------------------------------
    //   Obtain a psuedorandom real number in the range (0,1), i.e., a number
    //   greater than 0 and less than 1.
    inline mtprng::real_type mtprng::get_rand_real3()
    {
        // privides a granularity of approx. 2.3E-10
        return real_type((double(get_rand()) + 0.5) * (1.0 / 4294967296.0));
    }

} // end namespace CoyoteGulch

#endif
