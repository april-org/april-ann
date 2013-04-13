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

class table_of_token_codes {
public:
  // especiales 1024 (0x400)
  static const uint32_t error               = 0x0000;
  
  // notificaciones 1024
  // 1024-20
  static const uint32_t signal_end          = 0x1000;
  
  // tipos basicos 1024
  static const uint32_t boolean             = 0x2000;
  static const uint32_t token_char          = 0x2001;
  static const uint32_t token_int32         = 0x2002;
  static const uint32_t token_uint32        = 0x2003;
  
  // vectores:
  static const uint32_t vector_float        = 0x3000;
  static const uint32_t vector_double       = 0x3001;
  static const uint32_t vector_log_float    = 0x3002;
  static const uint32_t vector_log_double   = 0x3003;
  static const uint32_t vector_char         = 0x3004;
  static const uint32_t vector_int32        = 0x3005;
  static const uint32_t vector_uint32       = 0x3006;
  static const uint32_t vector_SymbolScores = 0x3007;
  
  // envio de grafos:
  static const uint32_t token_idag          = 0x4000;
  
};

#endif // TABLE_OF_TOKEN_CODES_H
