#include <cstdio>
#include "../c_src/logbase.h"

int main() {
  log_double a,b;
  int i=0;
  a = log_double::from_double(0);
  b = log_double::from_double(0.0001);
  printf("%4d rawvalue(%f) %f\n",i,a.log(),a.to_double());
  for (i=1; i<=10; i++) {
    a += b;
    printf("%4d rawvalue(%f) %f\n",i,a.log(),a.to_double());
  }

  log_float x = log_float::from_float(3.0f);
  log_double y= log_double::from_double(5.0);
  log_double z=x+y;
  printf("x=%f, y=%f, x+y=%f\n", x.to_float(), y.to_double(), z.to_double()); 
  //z=x-y;
  //printf("x=%f, y=%f, x-y=%f\n", x.to_float(), y.to_double(), z.to_double()); 
  z=x*y;
  printf("x=%f, y=%f, x*y=%f\n", x.to_float(), y.to_double(), z.to_double()); 
  z=x/y;
  printf("x=%f, y=%f, x/y=%f\n", x.to_float(), y.to_double(), z.to_double()); 
  return 0;
}
