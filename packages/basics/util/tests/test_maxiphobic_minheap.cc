#include "maxiphobic_minheap.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using std::rand;
using std::srand;

using namespace april_utils;

#define NUM_QUEUES 100000
#define NUM_ELEMENTS 10000000

int main()
{
	maxiphobic_minheap<int> h[NUM_QUEUES];

	srand(time(NULL));
	for (int i=0; i<NUM_ELEMENTS; i++)
	{
		int cola = rand()%NUM_QUEUES; 
		if (rand()%2 || h[cola].empty())
			h[cola].insert(rand());
		else
			h[cola].remove_min();
	}

	for (int i=0; i<NUM_QUEUES; i++)
	{
		while (!h[i].empty())
		{
			h[i].remove_min();
		}
	}

}
