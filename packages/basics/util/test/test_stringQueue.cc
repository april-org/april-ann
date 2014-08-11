#include <iostream>
#include "../c_src/stringQueue.cc"

int main() {
  StringQueue s;
  for (int i=0; i<10; ++i)
    s.printf("hola mundo %d\n",i);
  int t;
  std::cout << s.exportBuffer(t) << std::endl;
  std::cout << "ocupa: " << t << std::endl;
}
