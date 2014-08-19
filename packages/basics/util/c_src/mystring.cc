#include <cctype>
#include <cstring>
#include "mystring.h"

namespace april_utils {

  size_t strnspn(const char *buffer, const char *accept, size_t length) {
    if (length > 128) {
      char lookup_table[256];
      memset(lookup_table, 0, 256*sizeof(char));
      for (size_t i=0; accept[i] != '\0'; ++i) {
        lookup_table[static_cast<unsigned char>(accept[i])] = 1;
      }
      size_t pos=0;
      while(pos < length && buffer[pos] != '\0' &&
            lookup_table[static_cast<unsigned char>(buffer[pos])]) {
        ++pos;
      }
      return pos;
    }
    else {
      size_t pos=0;
      while(pos < length &&
            buffer[pos] != '\0' &&
            strchr(accept, buffer[pos])) {
        ++pos;
      }
      return pos;
    }
  }

  size_t strncspn(const char *buffer, const char *reject, size_t length) {
    if (length > 128) {
      char lookup_table[256];
      memset(lookup_table, 0, 256*sizeof(char));
      for (size_t i=0; reject[i] != '\0'; ++i) {
        lookup_table[static_cast<unsigned char>(reject[i])] = 1;
      }
      size_t pos=0;
      while(pos < length && buffer[pos] != '\0' &&
            !lookup_table[static_cast<unsigned char>(buffer[pos])]) {
        ++pos;
      }
      return pos;
    }
    else {
      size_t pos=0;
      while(pos < length &&
            buffer[pos] != '\0' &&
            !strchr(reject, buffer[pos])) {
        ++pos;
      }
      return pos;
    }
  }

  const char *strnchr(const char *buffer, int c, size_t length) {
    for (size_t i=0; i<length; ++i) {
      if (buffer[i] == c) return buffer + i;
      // WARNING this function looks the entire buffer if (buffer[i] == '\0') break;
    }
    return 0;
  }

  const char *strncchr(const char *buffer, int c, size_t length) {
    for (size_t i=0; i<length; ++i) {
      if (buffer[i] != c) return buffer + i;
      // WARNING this function looks the entire buffer if (buffer[i] == '\0') break;
    }
    return 0;
  }

  int strcmpi(const char *a, const char *b) {
    while(*a != '\0' && *b != '\0') {
      char lower_a = tolower(*a);
      char lower_b = tolower(*b);
      if (lower_a < lower_b) return -1;
      else if (lower_a > lower_b) return 1;
      ++a;
      ++b;
    }
    return 0;
  }

  const char *string::NULL_STRING = "\0";
    
  string::string() { }
  
  string::string(const char *ptr) : vec(ptr, ptr+strlen(ptr)+1) { }
  
  string::string(const char *ptr, size_t len) : vec(ptr, ptr+len) { }

  string::string(const string& other) : vec(other.vec) { }

  string::~string() { }

  string &string::operator=(const string &other) {
    vec = other.vec;
    return *this;
  }

  string &string::operator+=(const string &other) {
    append(other);
    return *this;
  }
  
  void string::append(const string &other) {
    for (size_type i=0; i<other.size(); ++i) vec.push_back(other[i]);
  }
  
  bool string::operator==(const string &other) const {
    if (size() != other.size()) return false;
    for (size_type i=0; i<size(); ++i) if (vec[i] != other.vec[i]) return false;
    return true;
  }
  
  bool string::operator<(const string &other) const {
    if (size() < other.size()) return true;
    if (size() > other.size()) return false;
    for (size_type i=0; i<size(); ++i) {
      if (vec[i] > other.vec[i]) return false;
      if (vec[i] < other.vec[i]) return true;
    }
    return false;
  }
  
  bool string::operator<=(const string &other) const {
    if (size() < other.size()) return true;
    if (size() > other.size()) return false;
    for (size_type i=0; i<size(); ++i) {
      if (vec[i] > other.vec[i]) return false;
      if (vec[i] < other.vec[i]) return true;
    }
    return true;
  }
  
  char &string::operator[](unsigned int i) { return vec[i]; }
  
  char  string::operator[](unsigned int i) const { return vec[i]; }
  
  void string::push_back(char c) { vec.push_back(c); }
  
  void string::pop_back() { vec.pop_back(); }
  
  const char *string::c_str() const {
    if (vec.empty()) return NULL_STRING;
    else return vec.begin();
  }
  
  const char *string::data() const { return vec.begin(); }
  
  char *string::begin() { return vec.begin(); }
  
  char *string::end() { return vec.end(); }

  const char *string::begin() const { return vec.begin(); }
  
  const char *string::end() const { return vec.end(); }
  
  string::size_type string::size() const { return vec.size(); }
  
  string::size_type string::max_size() const { return vec.max_size(); }
  
  void string::resize(string::size_type size) { vec.resize(size); }
  
  string::size_type string::capacity() const { return vec.capacity(); }
  
  void string::reserve(string::size_type size) { vec.reserve(size); }
  
  void string::clear() { vec.clear(); }
  
  bool string::empty() const { return vec.size() == 0; }
  
  void string::swap(string &other) { vec.swap(other.vec); }
  
  char  string::at(unsigned int i) const { return vec[i]; }
  
  char &string::at(unsigned int i) { return vec[i]; }
  
  char string::front() const { return vec[0]; }
  
  char &string::front() { return vec[0]; }
  
  char  string::back() const { return vec.back(); }
  
  char &string::back() { return vec.back(); }

  char *string::release() { return vec.release(); }
}
