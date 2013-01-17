#include <iostream>
using namespace std;
#include "qsort.h"

int main() { // Lectura de datos e invocación de la función de sort
  int i,leido,indDat;
  const int maxVec = 1024;
  int v[maxVec];
  cout << "Prueba de la función QuickSort\n"
       << "Introduce valores a ordenar (fin con ^D):\n";
  indDat=0;
  while (indDat < maxVec && cin >> leido) {
    v[indDat] = leido; indDat++;
  }
  april_utils::Sort(v,indDat); // ordenamos el vector
  cout << "El vector ordenado:\n";
  for(i=0; i < indDat; i++)
    cout << v[i] 
	 << (((i % 10 == 9) || (i == indDat-1)) ? '\n' : ' ');
}

