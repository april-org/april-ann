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
#include "unique_ptr.h"
#include "unused_variable.h"

namespace april_utils {
  /**
   * T can be any type, referenced or not referenced.
   */
  template<typename T>
  class WeakPtr {
  public:

    /**
     * Builds a WeakPtr from a given pointer, by default NULL.
     */
    WeakPtr(T *ptr=0) : ptr(ptr) { }
    WeakPtr(WeakPtr<T> &other) : ptr(other.ptr) { }
    WeakPtr(UniquePtr<T> &other) : ptr(other.ptr) { }
    WeakPtr(SharedPtr<T> &other) : ptr(other.ptr) { }
    
    /**
     * Destructor.
     */
    ~WeakPtr() { }
    
    /**
     * Dereferencing, returns the pointer itself.
     */
    T *operator->() { return ptr; }

    /**
     * Dereferencing, returns the pointer itself.
     */
    const T *operator->() const { return ptr; }

    /**
     * Dereferencing, returns a reference to the data.
     */
    T &operator*() { return *ptr; }

    /**
     * Dereferencing, returns a reference to the data.
     */
    const T &operator*() const { return *ptr; }
    
    WeakPtr<T> &operator=(WeakPtr<T> &other) {
      reset(other.get());
      return *this;
    }
    WeakPtr<T> &operator=(UniquePtr<T> &other) {
      reset(other.get());
      return *this;
    }
    WeakPtr<T> &operator=(SharedPtr<T> &other) {
      reset(other.get());
      return *this;
    }
    WeakPtr<T> &operator=(T *other) {
      reset(other);
      return *this;
    }
    
    /**
     * Bypasses the pointer, but stills having it.
     */
    T *get() {
      return ptr;
    }
    
    /**
     * Bypasses the pointer, but stills having it.
     */
    const T *get() const {
      return ptr;
    }
    
    /**
     * Releases the pointer and assigns it to NULL.
     */
    T *release() {
      T *tmp = ptr;
      ptr = 0;
      return tmp;
    }
    
    /**
     * Takes another pointer.
     */
    void take(T *other) {
      reset(other);
    }

    /**
     * Takes another pointer.
     */
    void reset(T *other=0) {
      ptr = other;
    }

    /**
     * Returns a SharedPtr which takes the ownership of the referenced pointer.
     */
    SharedPtr<T> lock() {
      return SharedPtr<T>(get());
    }
    
    bool empty() const {
      return ptr == 0;
    };
    
    operator bool() const {
      return !empty();
    }
    
  private:
    T *ptr;
  };
  
  template<typename T>
  WeakPtr<T> makeWeakPtr(T *ptr) {
    return WeakPtr<T>(ptr);
  }
  
} // namespace april_utils
 
#endif // WEAK_PTR_H
