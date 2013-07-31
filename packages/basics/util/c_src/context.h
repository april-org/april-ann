/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef CONTEXT_H
#define CONTEXT_H

#include "april_assert.h"
#include <cstdio>

/* april_utils::context<T>
 *
 * Una cola con acceso aleatorio a los elementos. Sirve para guardar el
 * contexto de los objetos de una secuencia que vamos recorriendo.
 *
 * Metodos publicos:
 *
 * - context(int before, int after): Constructor, crea un objeto de tipo
 *   context reservando espacio para (before) objetos de tipo T anteriores al
 *   elemento actual y (after) objetos de tipo T posteriores al actual.
 *
 *   +-----+-----+-----+-----+-----+-----+-----+-----+-----+
 *   | -4  | -3  | -2  | -1  |  0  | +1  | +2  | +3  | +4  |
 *   +-----+-----+-----+-----+-----+-----+-----+-----+-----+
 *
 *   <--------before-------->| cur |<---------after-------->
 *
 * - bool insert(T *elem): Toma un puntero a un objeto de tipo T y lo inserta
 *   en la posicion (after) del vector, dado que es el mas reciente,
 *   desplazando el resto y destruyendo, si procede, el mas antiguo.
 *   Devuelve true si el elemento se ha insertado, false si se intenta
 *   insertar despues de llamar a end_input()
 *
 * - bool insert(const T& elem): Igual que el anterior pero realizando una
 *   copia del objeto pasado como parámetro.
 *
 * - bool ready(): devuelve true cuando todos los elementos del contexto son
 *   validos. Al crear el context es necesario insertar (after+1) elementos
 *   para que ready() devuelva true. Al final del proceso ready() devolvera
 *   false cuando el elemento actual se encuentre despues del ultimo
 *   introducido.
 *   
 *   Ejemplo: entrada=ABCDE, before=1, after=2
 *
 *                      /--- Elemento actual
 *                      |
 *                      v
 *   t=1   insert(A)   A|A|--  ready=false
 *   t=2   insert(B)   A|A|B-  ready=false
 *   t=3   insert(C)   A|A|BC  ready=true
 *   t=4   insert(D)   A|B|CD  ready=true
 *   t=5   insert(E)   B|C|DE  ready=true
 *   ---   end_input()
 *   t=6   shift()     C|D|EE  ready=true
 *   t=7   shift()     D|E|EE  ready=true
 *   t=8   shift()     E|E|EE  ready=false
 *
 * - void end_input(): Se utiliza para marcar el final de los datos de entrada
 *
 * - bool shift(): Una vez marcado el final de la entrada con end_input(), cada
 *   llamada a shift() desplaza el vector. Conceptualmente es equivalente a
 *   insertar una copia del ultimo elemento de la entrada utilizando insert()
 *
 * - void reset(): Reinicializa el objeto. Borra todos los objetos del contexto
 *   y lo deja como si estuviera recien creado por el constructor.
 *
 * - const T& operator[]: Permite el acceso al contexto utilizando los indices indicados
 *   en el diagrama anterior. 0 representa el elemento actual. -1,-2... los
 *   elementos anteriores y 1,2,... los posteriores.
 */


namespace april_utils {

  template <typename T>
  class context {
    int before;
    int after;
    int num_inserts;
    int num_shifts; // Numero de desplazamientos tras el fin de la entrada
    bool end;
    bool fill_after;
    int vec_size;
    T** vec;

    void init(){
      num_inserts=0;
      num_shifts=0;
      end=false;
      fill_after=false;
    }

    void delete_contents() {
      // comprobamos que haya habido alguna insercion
      if (num_inserts != 0) {
	int before=0;
	int after=vec_size-1;
	
	while ((before < vec_size-1) && (vec[before] == vec[before+1]))
	  before++;
	
	// Si hay elementos nulos al final, no hay repetidos
	// posteriores y el bucle parara en el ultimo 0 empezando
	// desde el final del vector. No obstante, hacer un delete de un
	// puntero nulo es correcto (no hace nada) :)
	while ((after > before) && (vec[after] == vec[after-1]))
	  after--;
	
	for(int i=before; i<=after; i++) 
	  delete vec[i];
      }
      // ponemos los punteros del vector a 0
      for (int i = 0; i < vec_size; i++)
        vec[i] = 0;
    }

    public:
    context(int before, int after):
      before(before),
      after(after),
      vec_size(before+after+1)
    {
      // cuando comparamos un elemento y el siguiente del array
      // necesitamos al menos un array de tam = 2
      april_assert(before+after > 0); 
      
      vec = new T* [vec_size];
      
      init();
      for (int i = 0; i < vec_size; i++)
        vec[i] = 0;
    }

    ~context() {
      delete_contents();
      delete[] vec;
    }

    int get_before() const { return before; }
    int get_after()  const { return after; }

    // Toma un puntero y se adueña de el, liberandolo
    // cuando ya no sea necesario.
    bool insert(T* elem) {
      if (end) return false;

      if (before+num_inserts < vec_size) {
        // Aun no hemos llenado el vector por primera vez
        vec[before+num_inserts] = elem;
      } else {
        // El vector ya esta lleno, desplazamos e insertamos al final
        if (vec[0] != vec[1]) 
          delete vec[0];

        for (int i=0; i<vec_size-1; i++) {
          vec[i]=vec[i+1];
        }
        vec[vec_size-1] = elem;
      }
      
      // En caso de ser el primer elemento
      // insertamos punteros al primero como elementos
      // anteriores del contexto
      if (num_inserts==0) {
        for (int i=0; i<before; i++)
          vec[i] = vec[before];
      }
      

      num_inserts++;
      return true;
    }
    
    // Hace una copia e inserta un puntero a la copia
    bool insert(const T& elem) {
      if (end) return false;
      T* tmp=new T(elem);
      insert(tmp);
      return true;
    }

    bool is_ended() const { return end; }

    bool ready() const {
      if (!end)
        return (num_inserts > after);
      else
        return (num_shifts <= after) && (num_shifts <= num_inserts);
    }

    void end_input() {
      end = true;
      fill_after = (num_inserts > 0 && num_inserts < vec_size);
    }

    bool shift() {
      if (!end)
        return false;
      
      // Si no se ha llenado el vector, se replica el
      // ultimo elemento tantas veces como sea necesario
      // para llenar los elementos posteriores del contexto
      // y no se desplaza la primera vez (el elemento central
      // nunca ha estado ready y no ha sido accedido aun)
      if (fill_after) {
        for (int i=before+num_inserts; i<vec_size; i++)
          vec[i] = vec[before+num_inserts-1];
        fill_after = false;
        num_shifts++;
        return true;
      } else {
        if (vec[0] != vec[1]) 
          delete vec[0];

        for (int i=0; i<vec_size-1; i++)
          vec[i]=vec[i+1];

        num_shifts++;
        return true;
      }
    }

    void reset() {
      delete_contents();
      init();
    }

    // Devuelve el elemento en el instante t
    // -1, -2... representan los elementos anteriores
    // 1, 2... representan los elementos posteriores
    const T& operator[] (int t) const {
      // Condicion de ready para el caso de fin de entrada
      april_assert(!end || ((num_shifts <= after) && (num_shifts <= num_inserts)) );
      // Condicion de "ready" para el caso de acceso antes de llenar after
      april_assert(end || num_inserts > after || t <= num_inserts);
      // Comprobaciones de rango
      april_assert(t<=after);
      april_assert(t>=-before);
      return *vec[before+t];
    }    
  };

  // wrapper para un caso tipico de uso de context
  template <typename T>
  struct vec_wrapper {
    T *vec;
    int input_vec_size;
    vec_wrapper(T *v, int sz) : input_vec_size(sz) {
      vec = new T[sz];
      for (int i=0; i<input_vec_size; ++i) vec[i] = v[i];
    }
    vec_wrapper(const vec_wrapper &other) { // copy constructor
      input_vec_size = other.input_vec_size;
      vec = new T[input_vec_size];
      for (int i=0; i<input_vec_size; ++i) vec[i] = other.vec[i];
    }
    vec_wrapper& operator=(const vec_wrapper &other) {
      if (this != &other) {
	delete[] vec;
	input_vec_size = other.input_vec_size;
	vec = new T[input_vec_size];
	for (int i=0; i<input_vec_size; ++i) vec[i] = other.vec[i];
      }
      return (*this);
    }
    ~vec_wrapper() { delete[] vec; }
    T& operator[] (int index) const { return vec[index]; }
  };
  
  // wrapper para un caso tipico de uso de context
  template <typename T>
  class context_of_vectors {
    context<vec_wrapper<T> > ctxt;
    int input_vec_size;
  public:
    context_of_vectors(int before, int after, int input_vec_size):
      ctxt(before,after), input_vec_size(input_vec_size) {
    }

    ~context_of_vectors() {
    }

    bool insert(T* vec) {
      return ctxt.insert(new vec_wrapper<T>(vec,input_vec_size));
    }

    bool ready() const { return ctxt.ready(); }

    void end_input() { ctxt.end_input(); }

    bool is_ended() const { return ctxt.is_ended(); }

    bool shift() { return ctxt.shift(); }

    void reset() { ctxt.reset(); }

    T* getOutputVector() const {
      T *output = 0;
      if (ctxt.ready()) {
	int before = ctxt.get_before();
	int after  = ctxt.get_after();
	output = new T[(before+1+after)*input_vec_size];
	int o_idx = 0;
	for (int i=-before; i<=after; i++)
	  for(int j = 0; j < input_vec_size; j++, o_idx++)
	    output[o_idx] = ctxt[i][j];
      }
      return output;
    }

  };

}
#endif
