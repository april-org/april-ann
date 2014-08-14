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
#ifndef UNIQUE_PTR_H
#define UNIQUE_PTR_H

#include "april_assert.h"
#include "disallow_class_methods.h"
#include "error_print.h"
#include "ptr_ref.h"
#include "referenced.h"
#include "unused_variable.h"

namespace april_utils {
  
  /**
   * @brief Smart pointer for Referenced with sanity check of getRef()==1.
   *
   * T must be derivated from Referenced, but a safe check will be done to
   * ensure only one reference exists.
   */
  template< typename T,
            typename Referencer=DefaultReferencer<T>,
            typename Deleter=DefaultDeleter<T> >
  class UniquePtr {
    
    APRIL_DISALLOW_COPY_AND_ASSIGN(UniquePtr);
    
  public:
    
    /**
     * Builds a UniquePtr from a given pointer, by default NULL.
     */
    UniquePtr(T *ptr=0) : referencer(Referencer()), deleter(Deleter()),
                          ptr(ptr) {
      referencer(ptr);
      referencer.checkUnique(ptr);
    }
    
    /**
     * Builds a UniquePtr from other UniquePtr object, taking the ownership of
     * the referenced pointer.
     */
    UniquePtr(UniquePtr<T,Referencer,Deleter> &other) :
      referencer(Referencer()), deleter(Deleter()), ptr(other.release()) { }

    /**
     * Builds a UniquePtr from other UniquePtr object, taking the ownership of
     * the referenced pointer.
     */
    template<typename T1>
    UniquePtr(UniquePtr<T1,Referencer,Deleter> &other) :
      referencer(Referencer()), deleter(Deleter()), ptr(other.release()) { }

    /**
     * Resets the object to a NULL pointer, what will execute a DecRef.
     */
    ~UniquePtr() { reset(); }
    
    /**
     * Dereferencing, returns the pointer itself.
     */
    T *operator->() { april_assert(get() != 0); return get(); }

    /**
     * Dereferencing, returns a const reference to the pointer itself.
     */
    const T *operator->() const { april_assert(get() != 0); return get(); }
    
    /**
     * Dereferencing, returns a reference to the data.
     */
    T &operator*() { april_assert(get() != 0); return *get(); }
    
    /**
     * Dereferencing, returns a const reference to the data.
     */
    const T &operator*() const { april_assert(get() != 0); return *get(); }

    /**
     * Assignment operator, transfers the ownership of the pointer referenced by
     * the given other object.
     */
    UniquePtr<T,Referencer,Deleter> &operator=(UniquePtr<T,Referencer,Deleter> &other) {
      take(other.release());
      return *this;
    }

    /**
     * Assignment operator, transfers the ownership of the pointer referenced by
     * the given other object.
     */
    template<typename T1>
    UniquePtr<T,Referencer,Deleter> &operator=(UniquePtr<T1,Referencer,Deleter> &other) {
      take(other.release());
      return *this;
    }

    /**
     * Assignment operator, takes the ownership of the given pointer.
     */
    UniquePtr<T,Referencer,Deleter> &operator=(T *other) {
      reset(other);
      return *this;
    }
    
    /**
     * Operator[], returns a reference to the data.
     */
    T &operator[](int i) {
      april_assert(ptr != 0); 
      return ptr[i];
    }

    /**
     * Operator[], returns a reference to the data.
     */
    const T &operator[](int i) const {
      april_assert(ptr != 0); 
      return ptr[i];
    }

    bool operator==(const T *&other) const {
      return ptr == other;
    }

    /**
     * Bypasses the pointer, but stills having the ownership.
     */
    T *get() {
      return ptr;
    }

    /**
     * Bypasses the pointer, but stills having the ownership.
     */
    const T *get() const {
      return ptr;
    }
    
    /**
     * Releases the pointer and the ownership, but NOT executes DecRef, so it
     * transfers the ownership to the caller.
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
      referencer.checkUnique(ptr);
    }
    
    /**
     * DecRef its pointer, and takes ownership and IncRef the given pointer.
     *
     * @note By default receives a NULL pointer.
     */
    void reset(T *other = 0) {
      if (ptr != other) {
        referencer(other);
        deleter(ptr);
        ptr = other;
        referencer.checkUnique(ptr);
      }
    }
    
    bool empty() const {
      return get() == 0;
    };
    
  private:
    Referencer referencer;
    Deleter deleter;
    T *ptr;
  };
  
  template<typename T>
  UniquePtr<T> makeUniquePtr(T *ptr) {
    return UniquePtr<T>(ptr);
  }
  
} // namespace april_utils

#define STD_UNIQUE_PTR(T) april_utils::UniquePtr< T, april_utils::StandardReferencer<T>, april_utils::StandardDeleter<T> >
 
#endif // UNIQUE_PTR_H
