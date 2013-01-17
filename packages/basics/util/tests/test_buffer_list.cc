#include "../c_src/buffer_list.cc"
#include "../c_src/binarizer.cc"

#include <iostream>
using std::cout;
using std::endl;

int main() {
  buffer_list bl;

  for (int i=0;i<3;i++)
    bl.add_formatted_string_right("hola mundo! %d+%d=%d,",i,2,i+2);
  float v[10];
  for (int i=0;i<10;i++) v[i] = 0.81+i*0.1;
  bl.add_binarized_float_left (v, 10);

  uint32 blsize = bl.get_size();
  cout << "El bufferlist tiene talla: " << blsize << endl;

  // anyadimos el tamaño al inicio
  bl.add_binarized_uint32_left(&blsize,1);

  char *resul = bl.to_string(buffer_list::NULL_TERMINATED);
  cout << '"' << resul << '"' << endl;
  delete[] resul;
}

