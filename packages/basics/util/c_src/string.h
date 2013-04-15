#ifndef STRING_H
#define STRING_H

#include "vector.h"

namespace april_utils {

  class string {
  public:
    typedef char value_type;
    typedef vector<char> container;
    typedef char& reference;
    typedef const char& const_reference;
    typedef container::iterator        iterator;
    typedef container::const_iterator  const_iterator;
    typedef container::difference_type difference_type;
    typedef container::size_type       size_t;
  private:
    vector<char> vec;
    
  public:
    string();
    string(const char *ptr);
    string(const char *ptr, size_t len);
    string(const string& other);
    ~string();
    string &operator=(const string &other);
    string &operator+=(const string &other);
    void append(string &other);
    bool operator==(const string &other) const;
    bool operator<(const string &other)  const;
    bool operator<=(const string &other) const;
    char &operator[](unsigned int i);
    char  operator[](unsigned int i) const;
    void push_back(char c);
    void pop_back();
    const char *c_str() const;
    const char *data() const;
    char *begin();
    char *end();
    size_type size();
    size_type max_size();
    void resize(size_type size);
    size_type capacity();
    void reserve(size_type size);
    void clear();
    bool empty();
    void swap(string &other);
    char  at(unsigned int i) const;
    char &at(unsigned int i);
    char  front() const;
    char &front();
    char  back() const;
    char &back();
  };

}

#endif // STRING_H
