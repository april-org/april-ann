#include <iostream>
#include "constString.h"

using namespace std;

int main()
{
  constString s1("aaa");
  constString s2("aaaa");
  constString s3("aab");
  constString s4("aa");
  constString s5("aaa");

  cout << (const char *)(s1) << "==" << (const char *)(s2) << (s1 == s2) << endl;
  cout << (const char *)(s1) << "<" << (const char *)(s2) << (s1 < s2) << endl;
  cout << (const char *)(s1) << ">" << (const char *)(s2) << (s1 > s2) << endl;
  cout << (const char *)(s1) << "<=" << (const char *)(s2) << (s1 <= s2) << endl;
  cout << (const char *)(s1) << ">=" << (const char *)(s2) << (s1 >= s2) << endl;
  
  cout << (const char *)(s1) << "==" << (const char *)(s3) << (s1 == s3) << endl;
  cout << (const char *)(s1) << "<" << (const char *)(s3) << (s1 < s3) << endl;
  cout << (const char *)(s1) << ">" << (const char *)(s3) << (s1 > s3) << endl;
  cout << (const char *)(s1) << "<=" << (const char *)(s3) << (s1 <= s3) << endl;
  cout << (const char *)(s1) << ">=" << (const char *)(s3) << (s1 >= s3) << endl;
  
  cout << (const char *)(s1) << "==" << (const char *)(s4) << (s1 == s4) << endl;
  cout << (const char *)(s1) << "<" << (const char *)(s4) << (s1 < s4) << endl;
  cout << (const char *)(s1) << ">" << (const char *)(s4) << (s1 > s4) << endl;
  cout << (const char *)(s1) << "<=" << (const char *)(s4) << (s1 <= s4) << endl;
  cout << (const char *)(s1) << ">=" << (const char *)(s4) << (s1 >= s4) << endl;
  
  cout << (const char *)(s1) << "==" << (const char *)(s5) << (s1 == s5) << endl;
  cout << (const char *)(s1) << "<" << (const char *)(s5) << (s1 < s5) << endl;
  cout << (const char *)(s1) << ">" << (const char *)(s5) << (s1 > s5) << endl;
  cout << (const char *)(s1) << "<=" << (const char *)(s5) << (s1 <= s5) << endl;
  cout << (const char *)(s1) << ">=" << (const char *)(s5) << (s1 >= s5) << endl;
}

