#include "context.h"
#include <cstdio>

using april_utils::context;
using namespace std;

int main()
{
  context<int> c(3,3);
 
  c.insert(10);
  int *p = new int(20);
  c.insert(p);
  printf("p==%p\n", p);
  c.insert(30);
  if (!c.ready())
    printf("Not ready yet ;)\n");
  c.insert(40);
  printf("%d %d %d %d %d %d %d\n", c[-3],c[-2], c[-1], c[0], c[1], c[2], c[3]);
  c.insert(50);
  printf("%d %d %d %d %d %d %d\n", c[-3],c[-2], c[-1], c[0], c[1], c[2], c[3]);
  c.insert(60);
  printf("%d %d %d %d %d %d %d\n", c[-3],c[-2], c[-1], c[0], c[1], c[2], c[3]);
  c.insert(70);
  printf("%d %d %d %d %d %d %d\n", c[-3],c[-2], c[-1], c[0], c[1], c[2], c[3]);
  printf("END INPUT\n");
  c.end_input();
  c.shift();
  while (c.ready()) {
    printf("%d %d %d %d %d %d %d\n", c[-3],c[-2], c[-1], c[0], c[1], c[2], c[3]);
    c.shift();
  }
}
