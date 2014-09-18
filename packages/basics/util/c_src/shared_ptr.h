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

namespace AprilUtils {
  
  // forward declaration
  template<typename T> class WeakPtr;
  
  /**
   * @brief A smart pointer for shared references.
   *
   * @note T must be derivated from Referenced.
   */
  template< typename T,
            typename Referencer=DefaultReferencer<T>,
            typename Deleter=DefaultDeleter<T> >
  class SharedPtr {
    friend class WeakPtr<T>;

  public:
    
    /**
     * @brief Builds a SharedPtr from a given pointer, by default NULL.
     */
    SharedPtr(T *ptr=0) : referencer(Referencer()), deleter(Deleter()),
                          ptr(ptr) {
      referencer(ptr);
    }
    
    /**
     * @brief Builds a SharedPtr from other SharedPtr object, increasing the
     * reference counter.
     */
    SharedPtr(SharedPtr<T,Referencer,Deleter> &other) : 
      referencer(Referencer()), deleter(Deleter()), ptr(other.get()) {
      referencer(ptr);
    }

    /**
     * @brief Builds a SharedPtr from other SharedPtr object, increasing the
     * reference counter.
     */
    SharedPtr(const SharedPtr<T,Referencer,Deleter> &other) : 
      referencer(Referencer()), deleter(Deleter()), ptr(other.ptr) {
      referencer(ptr);
    }

    /**
     * @brief Builds a SharedPtr from other SharedPtr object, increasing the
     * reference counter.
     */
    template<typename T1>
    SharedPtr(SharedPtr<T1,Referencer,Deleter> &other) : 
      referencer(Referencer()), deleter(Deleter()), ptr(other.get()) {
      referencer(ptr);
    }

    /**
     * @brief Builds a SharedPtr from other SharedPtr object, increasing the
     * reference counter.
     */
    template<typename T1>
    SharedPtr(const SharedPtr<T1,Referencer,Deleter> &other) : 
      referencer(Referencer()), deleter(Deleter()), ptr(other.ptr) {
      referencer(ptr);
    }
    
    /**
     * @brief Resets the object to a NULL pointer, what will execute a DecRef.
     */
    ~SharedPtr() { reset(); }
    
    /**
     * @brief Dereferencing, returns the pointer itself.
     */
    T *operator->() { return get(); }

    /**
     * @brief Dereferencing, returns the pointer itself.
     */
    const T *operator->() const { return get(); }
    
    /**
     * @brief Dereferencing, returns a reference to the data.
     */
    T &operator*() { return *get(); }

    /**
     * @brief Dereferencing, returns a reference to the data.
     */
    const T &operator*() const { return *get(); }
    
    /**
     * @brief Assignment operator, copies the pointer and increases the
     * reference.
     */
    SharedPtr<T,Referencer,Deleter> &operator=(SharedPtr<T,Referencer,Deleter> &other) {
      reset(other.get());
      return *this;
    }

    /**
     * @brief Assignment operator, copies the pointer and increases the
     * reference.
     */
    SharedPtr<T,Referencer,Deleter> &operator=(const SharedPtr<T,Referencer,Deleter> &other) {
      reset(other.ptr);
      return *this;
    }

    /**
     * @brief Assignment operator, copies the pointer and increases the
     * reference.
     */
    template<typename T1>
    SharedPtr<T,Referencer,Deleter> &operator=(SharedPtr<T1,Referencer,Deleter> &other) {
      reset(other.get());
      return *this;
    }

    /**
     * @brief Assignment operator, copies the pointer and increases the
     * reference.
     */
    template<typename T1>
    SharedPtr<T,Referencer,Deleter> &operator=(const SharedPtr<T1,Referencer,Deleter> &other) {
      reset(other.ptr);
      return *this;
    }
    
    /**
     * @brief Assignment operator, copies the pointer and increases the
     * reference.
     */
    SharedPtr<T> &operator=(T *other) {
      reset(other);
      return *this;
    }
        
    bool operator==(const SharedPtr<T> &other) const {
      return ptr == other.ptr;
    }

    bool operator==(const T *&other) const {
      return ptr == other;
    }

    /**
     * @brief Bypasses the pointer, but stills having a reference.
     */
    T *get() {
      return ptr;
    }

    /**
     * @brief Bypasses the pointer, but stills having a reference.
     */
    const T *get() const {
      return ptr;
    }

    /**
     * @brief Releases the pointer, but NOT decreases its reference counter.
     */
    T *release() {
      T *tmp = ptr;
      ptr = 0;
      return tmp;
    }
    
    /**
     * @brief Takes the ownership without IncRef the pointer.
     */
    void take(T *other) {
      reset();
      ptr = other;
    }

    /**
     * @brief DecRef its pointer, and IncRef the given pointer.
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
    
  private:
    Referencer referencer;
    Deleter deleter;
    T *ptr;
  };
  
  template<typename T>
  SharedPtr<T> makeSharedPtr(T *ptr) {
    return SharedPtr<T>(ptr);
  }
  
} // namespace AprilUtils
 
#endif // SHARED_PTR_H
