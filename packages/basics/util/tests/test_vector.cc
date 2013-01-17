#include "vector.h"
#include <iostream>

using namespace april_utils;
using std::cout;
using std::endl;

int main()
{
    vector<int> v1;
    v1.push_back(1);
    v1.push_back(2);
    v1.push_back(3);
    v1.push_back(4);

    vector<int> v2;
    v2.push_back(30);
    v2.push_back(20);
    v2.push_back(10);

    v1.swap(v2);

    for (int i=0; i<v2.size(); ++i) {
        cout << v2[i] << endl;
    }

}
