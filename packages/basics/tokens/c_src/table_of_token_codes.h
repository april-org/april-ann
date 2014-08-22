/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#ifndef TABLE_OF_TOKEN_CODES_H
#define TABLE_OF_TOKEN_CODES_H

#include <stdint.h>

/*  TOKEN TYPE GROUPS PREPARED FOR FUTURE AMPLIATIONS

    Basic group with 4096 elements (0x1000)
   
    Block Number  Class
    0     0x0000  error types
    1     0x1000  signals
    2     0x2000  basic types
    3     0x3000  vectors
    4     0x4000  graphs
*/

namespace basics {

/// Definition of TokenCode type
typedef uint32_t TokenCode;

  /// Static class which contains TokenCodes for each corresponding Token type
  class table_of_token_codes {
  public:
    // especial codes 1024 (0x400)
    static const TokenCode error               = 0x0000;
  
    // notifications 1024
    // 1024-20
    static const TokenCode signal_end          = 0x1000;
  
    // basic types 1024
    static const TokenCode boolean             = 0x2000;
    static const TokenCode token_char          = 0x2001;
    static const TokenCode token_int32         = 0x2002;
    static const TokenCode token_uint32        = 0x2003;
    static const TokenCode token_mem_block     = 0x2004;
    static const TokenCode token_matrix        = 0x2005;
    static const TokenCode token_sparse_matrix = 0x2006;
    static const TokenCode token_null          = 0x2999;
  
    // vectors:
    static const TokenCode vector_float        = 0x3000;
    static const TokenCode vector_double       = 0x3001;
    static const TokenCode vector_log_float    = 0x3002;
    static const TokenCode vector_log_double   = 0x3003;
    static const TokenCode vector_char         = 0x3004;
    static const TokenCode vector_int32        = 0x3005;
    static const TokenCode vector_uint32       = 0x3006;
    static const TokenCode vector_SymbolScores = 0x3007;
    static const TokenCode vector_Tokens       = 0x3008;
  
    // graph protocol:
    static const TokenCode token_idag          = 0x4000;  
  };

} // namespace basics

#endif // TABLE_OF_TOKEN_CODES_H
