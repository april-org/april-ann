#include <iostream>
using namespace std;
#include "../c_src/fifo.h"
using namespace april_utils;

int main() {
  fifo<int> f;
  int i;
  char comando[10];
  while (cin >> comando) {
    if (strcmp(comando,"p") == 0) { // put
      cin >> i;
      f.put(i);
    }
    else if (strcmp(comando,"d") == 0) { // drop
      cin >> i;
      f.drop_by_value(i);
    }
    else if (strcmp(comando,"w") == 0) { // print
      while (f.get(i)) cout << i << " ";
      cout << endl;
    }
    else if (strcmp(comando,"g") == 0) { // get
      if (f.get(i)) {
	cout << "get devuelve valor " << i << endl;
      } else {
	cout << "get dice fifo vacía" << endl;
      }
    }
  }
  return 0;
}


