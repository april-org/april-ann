#include "slist.h"
#include <algorithm>
#include <numeric>
#include <iostream>

using april_utils::slist;
using std::cout;
using std::endl;

bool predicado(int x)
{
	return x>90;
}

void print_int(int i)
{
	cout << i << endl;
}

template<class T> void print(slist<T> l)
{
	for (typename slist<T>::iterator i = l.begin(); i != l.end(); i++)
		cout << *i << ", ";
	cout << endl;
}

template<class T> void print_const(const slist<T> l)
{
	// Si reemplazamos el const_iterator con un iterator aqui, falla
	// al compilar, porque l es const :D
	for (typename slist<T>::const_iterator i = l.begin(); i != l.end(); i++)
		cout << *i << ", ";
	cout << endl;
}

int main()
{
	slist<int> l;
	l.push_back(1);
	l.push_back(2);
	l.push_back(3);
	l.push_back(4);
	l.push_back(99);


	slist<int>::iterator it;
	slist<int>::const_iterator c_it = l.begin();

	cout << (l.begin() != l.end()) << endl;
	cout << (it != l.end()) << endl;
	cout << (c_it != l.end()) << endl;


	cout << "Recorremos con un iterator: ";
	for (slist<int>::iterator i = l.begin(); i != l.end(); i++)
		cout << *i << ", ";
	cout << endl;

	cout << "Recorremos con un const_iterator: ";
	for (slist<int>::const_iterator i = l.begin(); i != l.end(); i++)
		cout << *i << ", ";
	cout << endl;

	cout << "Recorrido con std::for_each: " << endl;
	std::for_each(l.begin(), l.end(), print_int);

	cout << "Reemplazamos el 3 por un 99: ";
	std::replace(l.begin(), l.end(), 3, 99);
	print(l);

	/*
	cout << "Rotate con middle = 2: "<< endl;
	std::rotate(l.begin(), ++l.begin(), l.end());
	print(l);

	cout << "La suma de los elementos de la lista es: "<< 
		std::accumulate(l.begin(), l.end(), 0) << endl;

	cout << "std::remove_if( x > 90 )" << endl;
	std::remove_if(l.begin(), l.end(), predicado);
	print(l);
	*/

	// TODO: Un test para slist::transfer_front_to_front()
	// TODO: Un test para slist::transfer_front_to_back()

}
