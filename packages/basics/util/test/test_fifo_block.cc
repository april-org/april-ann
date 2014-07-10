#include <iostream>
#include "../c_src/fifo_block.h"
using namespace std;
using namespace april_utils;

int main() {
  typedef fifo_block<int,7> fifo_block_type;
  fifo_block_type fifobint;
  int i,v;
  char comando[100];

  cout << "a ver si peta\n";
  fifo_block_type::iterator r = fifobint.begin();
  cout << fifobint.is_end(r) << endl;
  for (fifo_block_type::iterator r = fifobint.begin();
       !fifobint.is_end(r);
       ++r) {
    cout << *r << " ";
  }
  cout << "--a ver si peta\n";

  for (i=0; i<20; i++) fifobint.put(i);
  fifo_block_type una_copia(fifobint);
  cout << "La copia tiene " << una_copia.count() << " elementos\n";
  cout << "Aquí está la copia recorriendola con un   iterador ";
  for (fifo_block_type::iterator r = una_copia.begin();
       r != una_copia.end();
       ++r) {
    cout << *r << " ";
  }
  cout << "\nAquí está la copia recorriendola con otro iterador ";
  for (fifo_block_type::iterator r = una_copia.begin();
       !una_copia.is_end(r);
       ++r) {
    cout << *r << " ";
  }
  cout << "\nsacamos 5 valores de la cola original\n";
  for (i=0; i<5 && fifobint.get(v); i++) {
    cout << v << " ";
  }
  cout << "La original tiene " << fifobint.count() << " elementos\n";
  cout << "insertamos otros 5 valores del 1 al 5 en ambas colas\n";
  for (i=0; i<5 ; i++) {
    fifobint.put(i);
    una_copia.put(i);
  }
  cout << "asignamos una variable a otra\n";
  una_copia = fifobint;
  cout << "Aquí está la cola:\n";
  while (fifobint.get(v)) {
    cout << v << " ";
  }
  cout << "\nY la copia:\n";
  while (una_copia.get(v)) {
    cout << v << " ";
  }
  cout << endl;
  return 0;
}


