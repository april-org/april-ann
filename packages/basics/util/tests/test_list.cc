#define USE_ITERATOR_TAGS
#include "../c_src/list.h"

#include <algorithm>
#include <iostream>

using april_utils::list;
using std::cout;
using std::endl;

bool predicado(int x)
{
        return x>90;
}

void print_int(int i)
{
        cout << i << ", ";
}

template<class T> void print(list<T> l)
{
        for (typename list<T>::iterator i = l.begin(); i != l.end(); i++)
                cout << *i << ", ";
}

template<class T> void print_const(const list<T> l)
{
        // Si reemplazamos el const_iterator con un iterator aqui, falla
        // al compilar, porque l es const :D
        for (typename list<T>::const_iterator i = l.begin(); i != l.end(); i++)
                cout << *i << ", ";
}


int main()
{
  cout << "creando lista vacia" << endl;
  list<int> l;

  cout << "l.max_size() == " << l.max_size() << endl;
  cout << "l.size()  == " << l.size() << endl;
  cout << "l.empty() == " << l.empty() << endl;
 

  cout << "empujando elementos" << endl;
  l.push_back(4);
  l.push_back(5);
  l.push_back(6);
  l.push_back(7);
  l.push_front(3);
  l.push_front(2);
  l.push_front(1);
  l.push_back(8);
  l.push_back(9);
  l.push_front(0);

  cout << "l.size()  == " << l.size() << endl;
  cout << "l.empty() == " << l.empty() << endl;
  cout << "l.front() == " << l.front() << endl;
  cout << "l.back()  == " << l.back() << endl;
  
  cout << "Recorrido con std::for_each: ";
  std::for_each(l.begin(), l.end(), print_int);
  cout << endl;

  cout << "Recorrido inverso con std::for_each: ";
  std::for_each(l.rbegin(), l.rend(), print_int);
  cout << endl;

  cout << "Recorrido con iterator: ";
  print(l);
  cout << endl;

  cout << "Recorrido con const_iterator: ";
  print_const(l);
  cout << endl;

  cout << "creando lista l2 con 10 elementos por defecto" << endl;
  list<int> l2(10);
  cout << "l2:  ";
  std::for_each(l2.begin(), l2.end(), print_int);
  cout << endl;

  cout << "creando lista l3 con 10 elementos con valor -1234" << endl;
  list<int> l3(10, -1234);
  cout << "l3:  ";
  std::for_each(l3.begin(), l3.end(), print_int);
  cout << endl;

  cout << "creando lista l4 con elementos [4,8) de l" << endl;
  list<int>::iterator first=l.begin();
  list<int>::iterator last =l.end();

  for (int i=0; i<4; i++) first++;
  for (int i=0; i<2; i++) last--;
  list<int> l4(first,last);
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l.pop_front(): ";
  l.pop_front();
  std::for_each(l.begin(), l.end(), print_int);
  cout << endl;

  cout << "l.pop_back():  ";
  l.pop_back();
  std::for_each(l.begin(), l.end(), print_int);
  cout << endl;

  cout << "l.swap(l4)" << endl;
  l.swap(l4);
  cout << "l:  ";
  std::for_each(l.begin(), l.end(), print_int);
  cout << endl;
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.insert(++l4.begin(), l.begin(), l.end())" << endl;
  l4.insert(++l4.begin(), l.begin(), l.end());
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.insert(--l4.end(), 5, 999)" << endl;
  l4.insert(--l4.end(), 5, 999);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;
  
  cout << "l4.insert(l4.end(), -6789)" << endl;
  l4.insert(l4.end(), -6789);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "Erasing the second and third elements of l4" << endl;
  last = ++l4.begin();
  first = last++;
  ++last;
  l4.erase(first, last);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l3.clear()" << endl;
  l3.clear();
  cout << "l3:  ";
  std::for_each(l3.begin(), l3.end(), print_int);
  cout << endl;

  cout << "l4.resize(10)" << endl;
  l4.resize(10);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.resize(12, -1)" << endl;
  l4.resize(12, -1);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.splice(++l4.begin(), l)" << endl;
  l4.splice(++l4.begin(), l);
  cout << "l: ";
  std::for_each(l.begin(), l.end(), print_int);
  cout << endl;
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.remove(6)" << endl;
  l4.remove(6);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "l4.remove_if(x>90)" << endl;
  l4.remove_if(predicado);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << endl << "and now, some STL algorithms ;)" << endl << endl;
  cout << "l4.erase(std::find(l4.begin(), l4.end(), 7))" << endl;
  l4.erase(std::find(l4.begin(), l4.end(), 7));
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;


  list<int>::iterator i = std::find(l4.begin(), l4.end(), 5);
  l4.splice(l4.begin(), l4, i);
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;

  cout << "*l4.begin=99;" << endl;
  *l4.begin()=99;
  cout << "l4: ";
  std::for_each(l4.begin(), l4.end(), print_int);
  cout << endl;
}
