#include <iostream>
using namespace std;
#include "../c_src/qsort.h"

int main() { // Lectura de datos e invocación de la función de sort
  int i,leido,indDat;
  const int maxVec = 1024;
  int v[maxVec];
  cout << "Prueba de la función Selection\n"
       << "Introduce el valor k y luego los valores (fin con ^D):\n";
  int k;
  cin >> k;
  indDat=0;
  while (indDat < maxVec && cin >> leido) {
    v[indDat] = leido; indDat++;
  }
  int valor = april_utils::Selection(v,indDat,k-1);
  cout << "El " << k << "-esimo menor es " << valor << endl;
  cout << "Los " << k << " menores elementos (desordenados) son:\n";
  for (int i=0; i<k; ++i)
    cout << v[i] << endl;
}

