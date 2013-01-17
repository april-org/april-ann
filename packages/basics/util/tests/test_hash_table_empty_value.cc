#include "null_types.h"
#include "hash_table.h"
#include "aux_hash_table.h"
using namespace april_utils;

#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

typedef hash<const char *,NullType> hash_test;

int print_pair(hash_test::value_type p)
{
  cout << sizeof(p) << " " << sizeof(p.first) << " " << sizeof(p.second) << " " << p.first << "->" <<  endl;
}

int main() {
  hash_test a_table;

  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;  
  a_table["hola"]  = NullType();
  a_table["adios"] = NullType();
  a_table.insert("adios",NullType());
  a_table.erase("adios");
  cout << "a_table[adios] " 
       << (a_table.search("adios") ? "esta" : "no es encuentra")
       << endl;
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;

  cout << "Insertamos 5 elementos nuevos." << endl;
  a_table["uno"]    = NullType();
  a_table["dos"]    = NullType();
  a_table["tres"]   = NullType();
  a_table["cuatro"] = NullType();
  a_table["cinco"]  = NullType();
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;  

  cout << "Recorrido con un iterador:" << endl;
  for (hash_test::const_iterator i=a_table.begin(); i != a_table.end(); ++i) {
    cout << (*i).first << "->" << endl;
  }

  cout << "Y ahora con for_each" << endl;
  std::for_each(a_table.begin(), a_table.end(), print_pair);

  return 0;
}
