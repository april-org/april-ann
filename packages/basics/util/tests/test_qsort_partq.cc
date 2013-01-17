#include <iostream>
using namespace std;
#include "../c_src/qsort.h"

int main() { // Lectura de datos e invocación de la función de sort
  int i,leido,indDat;
  const int maxVec = 1024;
  int v[maxVec];
  cout << "Prueba de partq\n"
       << "Introduce valores (fin con ^D):\n";
  indDat=0;
  while (indDat < maxVec && cin >> leido) {
    v[indDat] = leido; indDat++;
  }
  april_utils::PartQ<int> queue;
  queue.configure(v,indDat);
  while (queue.extractMin(leido)) {
    cout << leido << endl;
  }
}

