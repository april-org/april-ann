
#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

#include "open_addressing_hash_timestamp.h"
#include "aux_hash_table.h"
using namespace april_utils;
using namespace hash_aux;

typedef open_addr_hash_timestamp<const char *,const char*> hash_test;

int print_pair(hash_test::value_type p)
{
  cout << p.first << "->" << p.second << endl;
}

bool predicado(hash_test::value_type p)
{
  return (!strcmp(p.second, "adios"));
}

int main() {
  hash_test a_table;

  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;  
  a_table["hola"]  = "hello";
  a_table["adios"] = "bye";
  a_table.insert("adios","ciao!");
  a_table["x"] = "adios";
  a_table["y"] = "adios";
  a_table["z"] = "adios";
  cout << "a_table[adios] = " << a_table["adios"] << endl;
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;
  cout << "Ahora borramos todas las entradas" << endl;
  a_table.clear();
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;
  a_table["hola"]  = "hi";
  a_table["adios"] = "ciao";
  cout << "a_table[adios] " 
       << (a_table.search("adios") ? "esta" : "no es encuentra")
       << endl;
  cout << "La tabla tiene " << a_table.size() << " entradas"
       << " y " << a_table.bucket_count() << " buckets" << endl;
  cout << "Insertamos elementos nuevos." << endl;
  a_table["uno"] = "1";
  a_table["dos"] = "2";
  a_table["tres"] = "3";
  a_table["cuatro"] = "4";
  a_table["cinco"] = "5";
  a_table["seis"] = "6";
  a_table["siete"] = "7";
  a_table["ocho"] = "8";
  a_table["nueve"] = "9";
  a_table["diez"] = "10";
  a_table["once"] = "11";
  a_table["doce"] = "12";
  cout << "La tabla tiene " << a_table.size() << " entradas"
       << " y " << a_table.bucket_count() << " buckets" << endl;

  cout << "Recorrido con un iterador:" << endl;
  for (hash_test::const_iterator i=a_table.begin(); i != a_table.end(); ++i) {
    cout << (*i).first << "->" << i->second << endl;
  }

  cout << "Y ahora con for_each" << endl;
  std::for_each(a_table.begin(), a_table.end(), print_pair);

  return 0;
}
