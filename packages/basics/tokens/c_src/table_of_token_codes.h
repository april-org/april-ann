#ifndef TABLE_OF_TOKEN_CODES_H
#define TABLE_OF_TOKEN_CODES_H

#include <stdint.h>

/* AGRUPAR TIPOS DE TOKEN EN FAMILIAS Y REPARTIR EL RANGO
   NUMERICO DE CARA A FUTURAS AMPLIACIONES
   
   Hacemos bloques basicos de 4096 (0x1000) elementos
   
   Bloque  Numero  Clase
   0     0x0000  tipos de error
   1     0x1000  senyales
   2     0x2000  tipos basicos
   3     0x3000  vectores
   4     0x4000  grafos
   

*/

typedef uint32_t TokenCode

class table_of_token_codes {
public:
  // especiales 1024 (0x400)
  static const TokenCode error               = 0x0000;
  
  // notificaciones 1024
  // 1024-20
  static const TokenCode signal_end          = 0x1000;
  
  // tipos basicos 1024
  static const TokenCode boolean             = 0x2000;
  static const TokenCode token_char          = 0x2001;
  static const TokenCode token_int32         = 0x2002;
  static const TokenCode token_uint32        = 0x2003;
  static const TokenCode token_mem_block     = 0x2004;
  
  // vectores:
  static const TokenCode vector_float        = 0x3000;
  static const TokenCode vector_double       = 0x3001;
  static const TokenCode vector_log_float    = 0x3002;
  static const TokenCode vector_log_double   = 0x3003;
  static const TokenCode vector_char         = 0x3004;
  static const TokenCode vector_int32        = 0x3005;
  static const TokenCode vector_uint32       = 0x3006;
  static const TokenCode vector_SymbolScores = 0x3007;
  static const TokenCode vector_Tokens       = 0x3008;
  
  // envio de grafos:
  static const TokenCode token_idag          = 0x4000;  
};

#endif // TABLE_OF_TOKEN_CODES_H
