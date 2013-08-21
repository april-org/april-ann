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
//BIND_HEADER_C

#include <errno.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <unistd.h>
#ifndef NO_OMP
#include <omp.h>
#endif
#include "omp_utils.h"

// copy paste de lualib.c
FILE **newfile (lua_State *L) {
  FILE **pf = (FILE **)lua_newuserdata(L, sizeof(FILE *));
  *pf = NULL;  /* file handle is currently `closed' */
  luaL_getmetatable(L, LUA_FILEHANDLE);
  lua_setmetatable(L, -2);
  return pf;
}

//BIND_END

//BIND_HEADER_H
#include "mfset.h"
#include "trie_vector.h"
#include "trie4lua.h"
#include "stopwatch.h"
#include "linear_least_squares.h"
#include "words_table.h"
#include <cmath>
#include <ctime>
#include "popen2.h"

using namespace april_utils;

//BIND_END

//BIND_FUNCTION util.version
{
  LUABIND_RETURN(int, APRILANN_VERSION_MAJOR);
  LUABIND_RETURN(int, APRILANN_VERSION_MINOR);
}
//BIND_END

//BIND_FUNCTION util.is_cuda_available
{
#ifdef USE_CUDA
  LUABIND_RETURN(bool, true);
#else
  LUABIND_RETURN(bool, false);
#endif
}
//BIND_END

//BIND_FUNCTION util.omp_set_num_threads
{
#ifndef NO_OMP
  int n;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, int, n);
  omp_set_num_threads(n);
#endif
}
//BIND_END

//BIND_FUNCTION util.omp_get_num_threads
{
  LUABIND_RETURN(int, omp_utils::get_num_threads());
}
//BIND_END

//BIND_FUNCTION util.gettimeofday
{
  LUABIND_CHECK_ARGN(==, 0);
  struct timeval tv;
  struct timezone tz;
  tz.tz_minuteswest=0;
  tz.tz_dsttime=0;
  if (gettimeofday(&tv, &tz) == -1) {
    LUABIND_FERROR1("Error in gettimeofday: %s",
		    strerror(errno));
  }
  LUABIND_RETURN(number,tv.tv_sec);
  LUABIND_RETURN(number,tv.tv_usec);
}
//BIND_END

//BIND_FUNCTION util.split_process
//DOC_BEGIN
// split_process(...)
/// como fork pero multiple, recibe numero de procesos y devuelve un
/// numero entre 1 y numero de procesos
//DOC_END
{
  const int TOPE = 100;
  int num_processes, which_I_am = 1;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_GET_PARAMETER(1, int, num_processes);
  if (num_processes < 1 || num_processes > TOPE)
    LUABIND_FERROR1("Error in split_process, number out of range: %d",
		    num_processes);    
  while (1) {
    if (which_I_am == num_processes) {
      // el ultimo hijo no devuelve PID, por que no tiene hijo
      // asociado
      LUABIND_RETURN(number,num_processes);
      break;
    }
    pid_t ch_pid=fork();
    if (ch_pid != 0) { // el padre
      LUABIND_RETURN(number,which_I_am);
      LUABIND_RETURN(number,ch_pid);
      break;
    } else { // el hijo
      which_I_am++;
    }
  }
}
//BIND_END

//BIND_FUNCTION util.wait
//DOC_BEGIN
// wait(...)
/// espera a que terminen TODOS los hijos del proceso
/// despues usar split_process
//DOC_END
{
  int status;
  wait(&status);
}
//BIND_END

//BIND_FUNCTION util.stdout_is_a_terminal
//DOC_BEGIN
// bool stdout_is_a_terminal()
/// returns true if stdout fd refers to a terminal
//DOC_END
{
    LUABIND_RETURN(bool, isatty(1));
}
//BIND_END

//BIND_FUNCTION io.popen2
//DOC_BEGIN
// popen2(...)
/// recibe una cadena con el comando, devuelve nil+msg_error o dos
/// descriptores de fichero
//DOC_END
{
  constString cs;
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  LUABIND_GET_PARAMETER(1,constString,cs);
  const char *command = (const char *)cs;
  bool iserror = false;
  int en; // = errno;
  int    infp,   outfp;
  FILE **fpin, **fpout;
  if (popen2(command, &infp, &outfp) < 0) {
    iserror=true; en=errno;
  } else {
    fpin  = newfile(L);
    fpout = newfile(L);
    *fpin = fdopen(infp , "w");
    if (*fpin==NULL)  { 
      iserror=true; en=errno; 
    } else {
      *fpout = fdopen(outfp, "r");
      if (*fpout==NULL) { iserror=true; en=errno; }
    }
  }
  if (iserror) {
    lua_pushnil(L);
    lua_pushfstring(L, "%s: %s", command, strerror(en));
  }
  return 2;
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME TrieVector util.trie_vector
//BIND_CPP_CLASS    TrieVector

//BIND_CONSTRUCTOR TrieVector
{
  int log_size;
  LUABIND_GET_PARAMETER(1, int, log_size);
  TrieVector *obj = new TrieVector(log_size);
  LUABIND_RETURN(TrieVector, obj);
}
//BIND_END

//BIND_METHOD TrieVector get_size
{
  LUABIND_RETURN(uint, obj->getSize());
}
//BIND_END

//BIND_METHOD TrieVector get_parent
{
  unsigned int node;
  LUABIND_GET_PARAMETER(1, uint, node);
  LUABIND_RETURN(uint, obj->getParent(node));
}
//BIND_END

//BIND_METHOD TrieVector get_word
{
  unsigned int node;
  LUABIND_GET_PARAMETER(1, uint, node);
  LUABIND_RETURN(uint, obj->getWord(node));
}
//BIND_END

//BIND_METHOD TrieVector has_child
{
  unsigned int node, word, destnode;
  LUABIND_GET_PARAMETER(1, uint, node);
  LUABIND_GET_PARAMETER(2, uint, word);
  bool ret = obj->hasChild(node, word, destnode);
  LUABIND_RETURN(bool, ret);
  LUABIND_RETURN(uint, destnode);
}
//BIND_END

//BIND_METHOD TrieVector get_child
{
  unsigned int node, word;
  LUABIND_GET_PARAMETER(1, uint, node);
  LUABIND_GET_PARAMETER(2, uint, word);
  LUABIND_RETURN(uint, obj->getChild(node, word));
}
//BIND_END

//BIND_METHOD TrieVector has_sequence
{
  unsigned int *sequence, destnode;
  int           length;
  bool          ret;
  LUABIND_TABLE_GETN(1, length);
  sequence = new unsigned int[length];
  LUABIND_TABLE_TO_VECTOR(1, uint, sequence, length);
  ret = obj->hasSequence(sequence, length, destnode);
  LUABIND_RETURN(bool, ret);
  LUABIND_RETURN(uint, destnode);
  delete[] sequence;
}
//BIND_END

//BIND_METHOD TrieVector search_sequence
{
  unsigned int *sequence;
  int           length;
  LUABIND_TABLE_GETN(1, length);
  sequence        = new unsigned int[length];
  LUABIND_TABLE_TO_VECTOR(1, uint, sequence, length);
  unsigned int id = obj->searchSequence(sequence, length);
  LUABIND_RETURN(uint, id);
  delete[] sequence;
}
//BIND_END

//BIND_METHOD TrieVector get_sequence
{
  unsigned int node;
  int maxlength;
  LUABIND_GET_PARAMETER(1, uint, node);
  LUABIND_GET_PARAMETER(2, int, maxlength);
  unsigned int *sequence = new unsigned int[maxlength];
  int ret = obj->getSequence(node, sequence, maxlength);
  if (ret >= 0) {
    LUABIND_VECTOR_TO_NEW_TABLE(uint, sequence, ret);
    LUABIND_RETURN_FROM_STACK(-1);
  }
  delete[] sequence;
}
//BIND_END
    
//BIND_METHOD TrieVector root_node
{
  LUABIND_RETURN(uint, obj->rootNode());
}
//BIND_END

//BIND_METHOD TrieVector clear
{
  obj->clear();
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME Trie4lua util.trie
//BIND_CPP_CLASS    Trie4lua

//BIND_CONSTRUCTOR Trie4lua
{
  LUABIND_RETURN(Trie4lua, new Trie4lua());
}
//BIND_END

//BIND_METHOD Trie4lua reserveId
{
  LUABIND_RETURN(int, obj->reserveId());
}
//BIND_END

//BIND_METHOD Trie4lua find
{
  int *sequence, length;
  LUABIND_TABLE_GETN(1, length);
  sequence = new int[length];
  LUABIND_TABLE_TO_VECTOR(1, int, sequence, length);
  LUABIND_RETURN(int, obj->find(sequence, length));
  delete[] sequence;
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME MFSet util.mfset
//BIND_CPP_CLASS MFSet

//BIND_CONSTRUCTOR MFSet
{
  obj = new MFSet();
  LUABIND_RETURN(MFSet, obj);
}
//BIND_END

//BIND_METHOD MFSet set_size
{
  int sz;
  LUABIND_GET_PARAMETER(1, int, sz);
  obj->setSize(sz);
}
//BIND_END

//BIND_DESTRUCTOR MFSet
{
}
//BIND_END

//BIND_METHOD MFSet clear
{
  obj->clear();
}
//BIND_END

//BIND_METHOD MFSet find
{
  LUABIND_CHECK_ARGN(==,1);
  int v;
  LUABIND_GET_PARAMETER(1, int, v);
  LUABIND_RETURN(int, obj->find(v-1)+1);
}
//BIND_END

//BIND_METHOD MFSet merge
{
  LUABIND_CHECK_ARGN(==,2);
  int v1, v2;
  LUABIND_GET_PARAMETER(1, int, v1);
  LUABIND_GET_PARAMETER(2, int, v2);
  obj->merge(v1-1, v2-1);
}
//BIND_END

//BIND_METHOD MFSet size
{
  LUABIND_RETURN(int, obj->size());
}
//BIND_END

//BIND_METHOD MFSet clone
{
  LUABIND_RETURN(MFSet, obj->clone());
}
//BIND_END

//BIND_METHOD MFSet print
{
  obj->print();
}
//BIND_END

//BIND_METHOD MFSet toString
{
  char *buffer;
  int size = obj->toString(&buffer);
  lua_pushlstring(L,buffer,size);
  delete[] buffer;
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD MFSet fromString
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  constString cs;
  LUABIND_GET_PARAMETER(1,constString,cs);
  obj->fromString(cs);
}
//BIND_END

//BIND_CLASS_METHOD MFSet fromString
{
  LUABIND_CHECK_ARGN(==, 1);
  LUABIND_CHECK_PARAMETER(1, string);
  constString cs;
  LUABIND_GET_PARAMETER(1,constString,cs);
  MFSet *obj = new MFSet();
  obj->fromString(cs);
  LUABIND_RETURN(MFSet, obj);
}
//BIND_END

/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME stopwatch util.stopwatch
//BIND_CPP_CLASS stopwatch

//BIND_CONSTRUCTOR stopwatch
//DOC_BEGIN
// stopwatch()
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(stopwatch, new april_utils::stopwatch());
}
//BIND_END

//BIND_DESTRUCTOR stopwatch
{
}
//BIND_END

//BIND_METHOD stopwatch reset
//DOC_BEGIN
// void resest()
/// inicializa el cronometro
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  obj->reset();
}
//BIND_END

//BIND_METHOD stopwatch stop
//DOC_BEGIN
// void stop()
/// para el cronometro
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  obj->stop();
}
//BIND_END

//BIND_METHOD stopwatch go
//DOC_BEGIN
// void go()
/// reinicia el cronometro
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  obj->go();
}
//BIND_END

//BIND_METHOD stopwatch read
//DOC_BEGIN
// double read()
/// lee cronometro
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(double, obj->read_cpu_time());
  LUABIND_RETURN(double, obj->read_wall_time());
}
//BIND_END

//BIND_METHOD stopwatch is_on
//DOC_BEGIN
// bool is_on()
/// devuelve true si esta en marcha, false si esta parado
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(bool, obj->is_on());
}
//BIND_END

//BIND_METHOD stopwatch clone
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(stopwatch, new april_utils::stopwatch(*obj));
}
//BIND_END

// FIXME: nanosleep puede volver antes, en tal caso para avisar a lua
// se podría devolver el booleano que devuelve (y que estamos
// ignorando) y el tiempo restante :P

//BIND_FUNCTION util.sleep
{
  LUABIND_CHECK_ARGN(==,1);
  double sleeptime;
  LUABIND_GET_PARAMETER(1, double, sleeptime);
  double seconds = floor(sleeptime);
  struct timespec req;
  req.tv_sec  = (time_t)seconds;
  req.tv_nsec = (long)((sleeptime-seconds)*1e6);
  nanosleep(&req, 0);
}
//BIND_END

//BIND_FUNCTION util.linear_least_squares
{
  LUABIND_CHECK_ARGN(==,1);
  int numPoints;
  LUABIND_TABLE_GETN(1, numPoints);
  double *x = new double[numPoints];
  double *y = new double[numPoints];
  for (int i=1; i <= numPoints; ++i) {
    lua_rawgeti(L,1,i); // punto i-esimo, es una tabla
    lua_rawgeti(L,-1,1); // x
    LUABIND_GET_PARAMETER(-1, double, x[i-1]);
    lua_pop(L,1); // la quitamos de la pila
    lua_rawgeti(L,-1,2); // y
    LUABIND_GET_PARAMETER(-1, double, y[i-1]);
    lua_pop(L,2); // la quitamos de la pila, tb la tabla
  }
  double a,b; // f(x) = a + b*x
  least_squares(x,y,numPoints,a,b);
  delete[] x;
  delete[] y;
  LUABIND_RETURN(double, a);
  LUABIND_RETURN(double, b);
}
//BIND_END


////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME WordsTable util.words_table
//BIND_CPP_CLASS    WordsTable

//BIND_CONSTRUCTOR WordsTable
{
  WordsTable *obj = new WordsTable();
  LUABIND_RETURN(WordsTable, obj);
}
//BIND_END

//BIND_METHOD WordsTable filter_words
{
  int size;
  unsigned int *t;
  LUABIND_TABLE_GETN(1, size);
  t    = new unsigned int[size+1];
  t[0] = 0;
  LUABIND_TABLE_TO_VECTOR(1, uint, (t+1), size);
  WordsTable *newobj = new WordsTable(obj, t);
  delete[] t;
  LUABIND_RETURN(WordsTable, newobj);
}
//BIND_END

//BIND_METHOD WordsTable insert_words
{
  unsigned int *vec, size;
  LUABIND_TABLE_GETN(1, size);
  vec = new unsigned int[size];
  LUABIND_TABLE_TO_VECTOR(1, uint, vec, size);
  obj->insertWords(vec, size);
  delete[] vec;
}
//BIND_END


//BIND_METHOD WordsTable get_words
{
  unsigned int *vec;
  unsigned int size, index;
  LUABIND_GET_PARAMETER(1, uint, index);
  size = obj->getWords(index, &vec);
  LUABIND_VECTOR_TO_NEW_TABLE(uint, vec, size);
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD WordsTable size
{
  LUABIND_RETURN(uint, obj->size());
}
//BIND_END

//////////////////////////////////////////////////////////////////////

