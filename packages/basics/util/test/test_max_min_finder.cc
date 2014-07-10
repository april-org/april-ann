#include <cstdio>
#include <cstdlib>
#include "../c_src/vector.h"
#include "../c_src/max_min_finder.h"

using namespace april_utils;

int main(int argc, char **argv) {
  vector<int> resultado;
  const int ci = 6;
  const int cd = 6;
  bool findmax = true;
  bool findmin = false;
  max_min_finder<int> finder(ci,cd,findmax,&resultado,findmin,0);
  int i;
  while (scanf("%d",&i) == 1)
    finder.put(i);
  finder.end_sequence();
  printf("Resultado:\n");  
  for (vector<int>::iterator r = resultado.begin();  r != resultado.end(); ++r)
    printf("%d\n",*r);
  return 0;
}

