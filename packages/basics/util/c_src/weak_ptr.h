/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
#ifndef WEAK_PTR_H
#define WEAK_PTR_H

#include "april_assert.h"
#include "error_print.h"
#include "referenced.h"
#include "shared_ptr.h"
#include "unused_variable.h"

namespace AprilUtils {
  /**
   * @brief A plain wrapper over a pointer.
   *
   * @note T can be any type, referenced or not referenced.
   *
   * The WeakPtr is constructed by passing a SharedPtr object.
   */
  template<typename T>
  class WeakPtr {
    
    WeakPtr(T *ptr=0) : ptr(ptr) { }

    WeakPtr<T> &operator=(T *other) {
      reset(other);
      return *this;
    }

    /**
     * @brief Takes another pointer.
     */
    void take(T *other) {
      reset(other);
    }

    /**
     * @brief Takes another pointer.
     */
    void reset(T *other=0) {
      ptr = other;
    }

  public:
    /**
     * @brief Builds a WeakPtr from a given pointer, by default NULL.
     */
    WeakPtr();
    WeakPtr(WeakPtr<T> &other) : ptr(other.ptr) { }
    template<typename R, typename D>
    WeakPtr(SharedPtr<T,R,D> &other) : ptr(other.ptr) { }
    WeakPtr(const WeakPtr<T> &other) : ptr(other.ptr) { }
    WeakPtr(const SharedPtr<T> &other) : ptr(other.ptr) { }

    template<typename T1>
    WeakPtr(WeakPtr<T1> &other) : ptr(other.ptr) { }
    template<typename T1, typename R, typename D>
    WeakPtr(SharedPtr<T1,R,D> &other) : ptr(other.ptr) { }
    template<typename T1>
    WeakPtr(const WeakPtr<T1> &other) : ptr(other.ptr) { }
    template<typename T1>
    WeakPtr(const SharedPtr<T1> &other) : ptr(other.ptr) { }
    
    /**
     * @brief Destructor.
     */
    ~WeakPtr() { }

    /// True if the pointer is not empty.
    operator bool() const { return !empty(); }
    
    /// True if the pointer is empty.
    bool operator !() const { return  empty(); }
    
    /**
     * @brief Dereferencing, returns the pointer itself.
     */
    T *operator->() { return ptr; }

    /**
     * @brief Dereferencing, returns the pointer itself.
     */
    const T *operator->() const { return ptr; }

    /**
     * @brief Dereferencing, returns a reference to the data.
     */
    T &operator*() { return *ptr; }

    /**
     * @brief Dereferencing, returns a reference to the data.
     */
    const T &operator*() const { return *ptr; }
    
    WeakPtr<T> &operator=(WeakPtr<T> &other) {
      reset(other.get());
      return *this;
    }
    WeakPtr<T> &operator=(SharedPtr<T> &other) {
      reset(other.get());
      return *this;
    }
    WeakPtr<T> &operator=(const WeakPtr<T> &other) {
      reset(other.ptr);
      return *this;
    }
    
    template<typename T1>
    WeakPtr<T> &operator=(WeakPtr<T1> &other) {
      reset(other.get());
      return *this;
    }
    template<typename T1,typename R,typename D>
    WeakPtr<T> &operator=(SharedPtr<T1,R,D> &other) {
      reset(other.get());
      return *this;
    }
    template<typename T1>
    WeakPtr<T> &operator=(const WeakPtr<T1> &other) {
      reset(other.ptr);
      return *this;
    }
    template<typename T1>
    WeakPtr<T> &operator=(const SharedPtr<T1> &other) {
      reset(other.ptr);
      return *this;
    }

    /**
     * @brief Operator[], returns a reference to the data.
     */
    T &operator[](int i) {
      return ptr[i];
    }

    /**
     * @brief Operator[], returns a reference to the data.
     */
    const T &operator[](int i) const {
      return ptr[i];
    }

    
    bool operator==(const WeakPtr<T> &other) const {
      return ptr == other.ptr;
    }
    
    bool operator==(const SharedPtr<T> &other) const {
      return ptr == other.ptr;
    }
    
    bool operator==(const T *&other) const {
      return ptr == other;
    }
    
    /**
     * @brief Bypasses the pointer, but stills having it.
     */
    T *get() {
      return ptr;
    }
    
    /**
     * @brief Bypasses the pointer, but stills having it.
     */
    const T *get() const {
      return ptr;
    }
    
    /**
     * @brief Releases the pointer and assigns it to NULL.
     */
    T *release() {
      T *tmp = ptr;
      ptr = 0;
      return tmp;
    }
    
    /**
     * @brief Returns a SharedPtr which takes the ownership of the referenced
     * pointer.
     */
    SharedPtr<T> lock() {
      return SharedPtr<T>(get());
    }
    
    bool empty() const {
      return ptr == 0;
    };
        
  private:
    T *ptr;
  };
  
  template<typename T>
  WeakPtr<T> makeWeakPtr(T *ptr) {
    return WeakPtr<T>(ptr);
  }
  
} // namespace AprilUtils
 
#endif // WEAK_PTR_H
