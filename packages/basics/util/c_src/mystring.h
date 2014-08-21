#ifndef STRING_H
#define STRING_H

#include <cstring>
#include "vector.h"

namespace april_utils {

  size_t strnspn(const char *buffer, const char *accept, size_t length);
  size_t strncspn(const char *buffer, const char *reject, size_t length);
  const char *strnchr(const char *buffer, int c, size_t length);
  const char *strncchr(const char *buffer, int c, size_t length);
  int strcmpi(const char *a, const char *b);
  
  class string {
    static const char *NULL_STRING;
  public:
    typedef char value_type;
    typedef vector<char> container;
    typedef char& reference;
    typedef const char& const_reference;
    typedef container::iterator        iterator;
    typedef container::const_iterator  const_iterator;
    typedef container::difference_type difference_type;
    typedef container::size_type       size_type;
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
    void append(const string &other);
    operator const char *() const { return c_str(); }
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
    const char *begin() const;
    const char *end() const;
    size_type len() const;
    size_type size() const;
    size_type max_size() const;
    void resize(size_type size);
    size_type capacity() const;
    void reserve(size_type size);
    void clear();
    bool empty() const;
    void swap(string &other);
    char  at(unsigned int i) const;
    char &at(unsigned int i);
    char  front() const;
    char &front();
    char  back() const;
    char &back();
    char *release();
  };
  
}

#endif // STRING_H
