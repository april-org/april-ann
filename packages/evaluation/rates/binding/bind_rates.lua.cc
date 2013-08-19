//BIND_HEADER_C

// FIXME MACROS ELIMINAR ESTA FUNCION DE AQUI

int_sequence read_int_sequence(lua_State *L) {
  // recibe una tabla en el tope de la pila y la consume
  // stack: vector-table]
  // meter los elementos en un vector de enteros
  int i,j,k;
  int maxlength = 1024;
  int *aux_seq = new int[maxlength];
  // stack: j-component table
  k = 0;
  for (j=1;
       lua_rawgeti(L, -1, j), !lua_isnil(L,-1);
       j++) {
    // stack: j-component table

    if (j == maxlength) { // resize
      int *aux = new int[2*maxlength];
      for (i=0;i<maxlength;i++) 
	aux[i] = aux_seq[i];
      delete[] aux_seq; aux_seq = aux;
      maxlength *= 2;
    }

    aux_seq[k] = (int)luaL_checknumber(L, -1);
    lua_pop(L,1); // stack: table
    k++;
  }
  // stack: nil table
  lua_pop(L,2); // delete nil value and table
  int_sequence resul;
  resul.size   = k;
  resul.symbol = new int[k];
  for (j=0;j<k;j++) 
    resul.symbol[j] = aux_seq[j];
  delete[] aux_seq;
  return resul;
}

// FIXME MACROS ELIMINAR ESTA FUNCION DE AQUI????

pairs_int_sequences* read_pairs_int_sequences(lua_State *L) {

  // stack: data-table]

  pairs_int_sequences *data = 0;
  pairs_int_sequences **tail_data = &data;
  pairs_int_sequences *nextpair;

  for (int i=1;
       lua_rawgeti(L, -1, i), !lua_isnil(L,-1);
       i++) {
    // stack: i-pair-table data-table]

    // desplegar la tabla en el tope de la pila
    
    // creamos un nuevo par de secuencias
    nextpair = new pairs_int_sequences;
    nextpair->next = 0;

    // primera entrada
    lua_rawgeti(L,-1,1);
    // stack: 1st-component i-pair-table data-table]
    nextpair->correct = read_int_sequence(L);
    // stack: i-pair-table data-table]

    // segunda entrada
    lua_rawgeti(L,-1,2);
    // stack: 2nd-component i-pair-table data-table]
    nextpair->test = read_int_sequence(L);
    // stack: i-pair-table data-table]

    // añadir nextpair en la lista "data", al final
    *tail_data = nextpair;
    tail_data = &(nextpair->next);

    // nos cargamos la ref. al par de tablas que acabamos de read
    lua_pop(L,1);
    // stack: data-table]
  }
  lua_pop(L,2); // delete nil value and data-table
    // stack: ]
  return data;
}

// FIXME posiblemente estaria mejor en otra parte:

void delete_pairs_int_sequences (pairs_int_sequences *data) {
  pairs_int_sequences *aux;
  while (data != 0) {
    aux = data;
    data = data->next;
    delete[] aux->correct.symbol;
    delete[] aux->test.symbol;
    delete aux;
  }
}
//BIND_END

//BIND_HEADER_H
#include "rates.h"
//BIND_END

//BIND_FUNCTION rates.ints
{
  // recibe como argumentos: 
  // la tabla con los ints, 
  // el tipo de tasa
  // el valor de p, <- opcional
  // un valor booleano "with_matrix" <- opcional
  // devuelve:
  // el valor de la tasa
  // la tabla con la matriz, en caso de pedirse
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  // FIXME MACROS 
  lua_pushstring(L, "int_data");
  lua_gettable(L,1);
  if (lua_isnil(L,-1)) {
    lua_pushstring(L,"rates: int_data field not found");
    lua_error(L);
  }
  pairs_int_sequences *data = read_pairs_int_sequences(L);
  bool with_p = false,with_matrix=false;
  // FIXME MACROS 
  double p=1.0;
  lua_pushstring(L, "p");
  lua_gettable(L,1);
  if (!lua_isnil(L,-1)) {
    with_p = true;
    p = luaL_checknumber(L, -1);
    lua_pop(L,1);
  }
  lua_pushstring(L, "confusion_matrix");
  lua_gettable(L,1);
  if (lua_isboolean(L,-1)) {
    with_matrix = (bool)lua_toboolean(L,-1);
    lua_pop(L,1);
  }
  lua_pushstring(L, "rate");
  lua_gettable(L,1);
  if (lua_isnil(L,-1)) {
    lua_pushstring(L,"rates: rate field not found");
    lua_error(L);
  }
  const char *typerate = lua_tostring(L,-1);
  // stack: typerate table]
  counter_edition counted;
  conf_matrix *m;
  double rrate = rates::rate(data, typerate, counted, p, with_p,
			     (with_matrix) ? (&m) : (0));
  delete_pairs_int_sequences(data);
  // valores devueltos
  lua_newtable(L);
  lua_pushstring(L,"rate");
  lua_pushnumber(L, rrate);
  lua_settable(L,-3);
  lua_pushstring(L,"p");
  lua_pushnumber(L, p);
  lua_settable(L,-3);
  lua_pushstring(L,"sust");
  lua_pushnumber(L, counted.ns);
  lua_settable(L,-3);
  lua_pushstring(L,"ins");
  lua_pushnumber(L, counted.ni);
  lua_settable(L,-3);
  lua_pushstring(L,"borr");
  lua_pushnumber(L, counted.nb);
  lua_settable(L,-3);
  lua_pushstring(L,"ac");
  lua_pushnumber(L, counted.na);
  lua_settable(L,-3);
  if (with_matrix) {
    lua_pushstring(L,"confusion_matrix");
    lua_newtable(L);
    // rellenar matriz
    lua_pushstring(L, "confusion_matrix_dictionary");
    lua_gettable(L,1);
    // stack: dictionary confusion_matrix]
    int dsize = m->columns;
    bool con_diccionario = !lua_isnil(L,-1);
    for (int i=0;i<dsize;i++) {
      lua_pushnumber(L, i);
      if (con_diccionario) lua_gettable(L,-2);
      // stack: nombrefila dictionary confusion_matrix ...]
      lua_newtable(L);
      // stack: fila nombrefila dictionary confusion_matrix ...]
      for (int j=0;j<dsize;j++) {
	int v= (*m)[i][j];
	if (v != 0) {
	  lua_pushnumber(L, j);
	  // stack: idcolumna fila nombrefila dictionary confusion_matrix ...]
	  if (con_diccionario) lua_gettable(L,-4);
	  // stack: nombrecolumna fila nombrefila dictionary confusion_matrix ...]
	  lua_pushnumber(L, v);
	  // stack: valor nombrecolumna fila nombrefila dictionary confusion_matrix ...]
	  lua_settable(L,-3);
	  // stack: fila nombrefila dictionary confusion_matrix ...]
	}
      }
      // stack: fila nombrefila dictionary confusion_matrix ...]
      lua_settable(L,-4);
      // stack: dictionary confusion_matrix "confusion_matrix"]
    }
    lua_pop(L,1);
    // stack: confusion_matrix "confusion_matrix" result_table]
    lua_settable(L,-3);

    delete m;
  }
  return 1;
}
//BIND_END

//BIND_FUNCTION rates.raw
{
  // recibe como argumentos: 
  // la tabla con los ints, 
  // el valor de p, <- opcional
  // devuelve:
  // vector con tablas con na,ns,ni,nb;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, table);
  // FIXME MACROS 
  lua_pushstring(L, "int_data");
  lua_gettable(L,1);
  if (lua_isnil(L,-1)) {
    lua_pushstring(L,"rates: int_data field not found");
    lua_error(L);
  }
  pairs_int_sequences *data = read_pairs_int_sequences(L);
  bool with_p = false;
  // FIXME MACROS 
  double p=1.0;
  lua_pushstring(L, "p");
  lua_gettable(L,1);
  if (!lua_isnil(L,-1)) {
    with_p = true;
    p = luaL_checknumber(L, -1);
    lua_pop(L,1);
  }
  // tabla a devolver
  lua_newtable(L);
  int index = 1;
  for (const pairs_int_sequences *r = data; r != 0; r = r->next) {
    counter_edition counted = rates::gp(p, r->correct, r->test);
    lua_newtable(L);
    lua_pushnumber(L,counted.na);
    lua_setfield(L,-2,"na");
    lua_pushnumber(L,counted.ns);
    lua_setfield(L,-2,"ns");
    lua_pushnumber(L,counted.ni);
    lua_setfield(L,-2,"ni");
    lua_pushnumber(L,counted.nb);
    lua_setfield(L,-2,"nb");
    //
    lua_rawseti(L,-2,index); index++;
  }
  delete_pairs_int_sequences(data);
  return 1;
}
//BIND_END

