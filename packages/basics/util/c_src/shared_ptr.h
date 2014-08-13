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
#ifndef SHARED_PTR_H
#define SHARED_PTR_H

#include "april_assert.h"
#include "error_print.h"
#include "ptr_ref.h"
#include "referenced.h"
#include "unused_variable.h"

namespace april_utils {
  /**
   * T must be derivated from Referenced.
   */
  template< typename T,
            typename Referencer=DefaultReferencer<T>,
            typename Deleter=DefaultDeleter<T> >
  class SharedPtr {
  public:
    
    /**
     * Builds a SharedPtr from a given pointer, by default NULL.
     */
    SharedPtr(T *ptr=0) : referencer(Referencer()), deleter(Deleter()),
                          ptr(ptr) {
      referencer(ptr);
    }
    
    /**
     * Builds a SharedPtr from other SharedPtr object, increasing the
     * reference counter.
     */
    SharedPtr(SharedPtr<T,Referencer,Deleter> &other) : 
      referencer(Referencer()), deleter(Deleter()), ptr(other.get()) {
      referencer(ptr);
    }
    
    /**
     * Resets the object to a NULL pointer, what will execute a DecRef.
     */
    ~SharedPtr() { reset(); }
    
    /**
     * Dereferencing, returns the pointer itself.
     */
    T *operator->() { return get(); }

    /**
     * Dereferencing, returns the pointer itself.
     */
    const T *operator->() const { return get; }
    
    /**
     * Dereferencing, returns a reference to the data.
     */
    T &operator*() { return *get(); }
    const T &operator*() const { return *get(); }
    
    /**
     * Assignment operator, copies the pointer and increases the reference.
     */
    SharedPtr<T,Referencer,Deleter> &operator=(SharedPtr<T,Referencer,Deleter> &other) {
      reset(other.get());
      return *this;
    }
    
    /**
     * Assignment operator, copies the pointer and increases the reference.
     */
    SharedPtr<T> &operator=(T *other) {
      reset(other);
      return *this;
    }
    
    /**
     * Bypasses the pointer, but stills having a reference.
     */
    T *get() {
      return ptr;
    }

    /**
     * Bypasses the pointer, but stills having a reference.
     */
    const T *get() const {
      return ptr;
    }

    /**
     * Releases the pointer, but NOT decreases its reference counter.
     */
    T *release() {
      T *tmp = ptr;
      ptr = 0;
      return tmp;
    }
    
    /**
     * Takes the ownership without IncRef the pointer.
     */
    void take(T *other) {
      reset();
      ptr = other;
    }

    /**
     * DecRef its pointer, and IncRef the given pointer.
     *
     * @note By default receives a NULL pointer.
     */
    void reset(T *other = 0) {
      if (ptr != other) {
        referencer(other);
        deleter(ptr);
        ptr = other;
      }
    }
    
    bool empty() const {
      return get() == 0;
    }

    operator bool() const {
      return !empty();
    }
    
  private:
    Referencer referencer;
    Deleter deleter;
    T *ptr;
  };
  
  template<typename T>
  SharedPtr<T> makeSharedPtr(T *ptr) {
    return SharedPtr<T>(ptr);
  }
  
} // namespace april_utils
 
#endif // SHARED_PTR_H
