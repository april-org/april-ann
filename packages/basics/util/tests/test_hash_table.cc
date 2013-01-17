#include "hash_table.h"
#include "aux_hash_table.h"
using namespace april_utils;

#include <iostream>
#include <algorithm>
using std::cout;
using std::endl;

typedef hash<const char *,const char*> hash_test;

int print_pair(hash_test::value_type p)
{
  cout << p.first << "->" << p.second << endl;
}

bool predicado(hash_test::value_type p)
{
  return (!strcmp(p.second, "adios"));
}

struct X
{
  int data;

  bool operator == (const X& other) const
  {
    return data == other.data;
  }
};

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
  a_table.erase("adios");
  cout << "a_table[adios] " 
       << (a_table.search("adios") ? "esta" : "no es encuentra")
       << endl;
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;


  cout << "Insertamos 5 elementos nuevos." << endl;
  a_table["uno"] = "1";
  a_table["dos"] = "2";
  a_table["tres"] = "3";
  a_table["cuatro"] = "4";
  a_table["cinco"] = "5";
  cout << "La tabla tiene " << a_table.size() << " entradas" << endl;  

  cout << "Recorrido con un iterador:" << endl;
  for (hash_test::const_iterator i=a_table.begin(); i != a_table.end(); ++i) {
    cout << (*i).first << "->" << i->second << endl;
  }

  cout << "Borramos todas las entradas con valor 'adios'" << endl;
  a_table.delete_if(predicado);

  cout << "Redimensionamos la tabla" << endl;
  a_table.resize(128);

  cout << "Y ahora con for_each" << endl;
  std::for_each(a_table.begin(), a_table.end(), print_pair);

  X a,b;
  a.data=3;
  b.data=3;

  cout << "Test con tabla con HashFcn y EqualKey genericas" << endl;
  april_utils::hash<X, int> t;
  t[a] = 4;
  t[b] = 99;

  for (april_utils::hash<X,int>::iterator i=t.begin(); i!=t.end(); i++)
    cout << i->first.data << "->" << i->second << endl;;

  return 0;
}
