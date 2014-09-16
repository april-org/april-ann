/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#ifndef LIST_H
#define LIST_H

#include <stddef.h>
#include "swap.h"

#ifdef USE_ITERATOR_TAGS
#include <iterator>
#endif

namespace AprilUtils {

  namespace list_aux_templates{
    class _true_type {};
    class _false_type {};
    template<typename X> struct is_integer{ typedef _false_type type; };
    template<> struct is_integer<signed char>       { typedef _true_type type; };
    template<> struct is_integer<signed short>      { typedef _true_type type; };
    template<> struct is_integer<signed int>        { typedef _true_type type; };
    template<> struct is_integer<signed long>       { typedef _true_type type; };
    template<> struct is_integer<signed long long>  { typedef _true_type type; };
    template<> struct is_integer<unsigned char>     { typedef _true_type type; };
    template<> struct is_integer<unsigned short>    { typedef _true_type type; };
    template<> struct is_integer<unsigned int>      { typedef _true_type type; };
    template<> struct is_integer<unsigned long>     { typedef _true_type type; };
    template<> struct is_integer<unsigned long long>{ typedef _true_type type; };
  }

  // Be careful! It's not std::list :)
  // Implements Reversible Container, Front Insertion Sequence and Back Insertion Sequence
  template<class T> class list {
    // Linked-list nodes
    struct node {
      T     data;
      node *prev;
      node *next;
      // Constructor
      node(): data(T()), prev(this), next(this) {}
      node(T t, node* p, node *n):data(t), prev(p), next(n) {}
    };

    // Circular list implementation
    node sentinel;
    size_t list_size;

    void init_list() {
      sentinel.prev = &sentinel;
      sentinel.next = &sentinel;
      list_size = 0; 
    }

    // Some template magic in order to distinguish list(iterator, iterator) from list(int, int)
    // in template<class InputIterator> list(InputIterator, InputIterator)


    public:
    typedef T         value_type;      //< The type of object, T, stored in the list.
    typedef T*        pointer;         //< Pointer to T.
    typedef T&        reference;       //< Reference to T.
    typedef const T&  const_reference; //< Const reference to T
    typedef size_t    size_type;       //< An unsigned integral type.
    typedef ptrdiff_t difference_type; //< A signed integral type.

    // Bidirectional containers must define iterator, const_iterator AND
    // reverse_iterator, const_reverse_iterator.
    // FIXME: Equality comparison between iterator and const_iterator is broken

    /// Iterator used to iterate through a list.
    struct iterator { 
      friend class list<T>;
      //private:
        node *ptr;

      public:
        typedef T         value_type;
        typedef T*        pointer;
        typedef T&        reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        #ifdef USE_ITERATOR_TAGS
        typedef std::bidirectional_iterator_tag iterator_category;
        #endif

        iterator(): ptr(0) {}
        iterator(void *p): ptr(static_cast<node*>(p)) {}

        iterator& operator++() { // preincrement
          ptr = ptr->next;
          return *this;
        }

        iterator operator++(int) { // postincrement
          node *tmp = ptr;
          ptr = ptr->next;
          return iterator(tmp);
        }
        
        iterator& operator--() { // predecrement
          ptr = ptr->prev;
          return *this;
        }

        iterator operator--(int) { // postdecrement
          node *tmp = ptr;
          ptr = ptr->prev;
          return iterator(tmp);
        }

        T& operator *() const { // dereference
          return ptr->data;
        }

        T* operator ->() const {
          return &(ptr->data);
        }

        bool operator == (const iterator &i) const { return ptr == i.ptr; }
        bool operator != (const iterator &i) const { return ptr != i.ptr; }

    };

    /// Const iterator used to iterate through a list. 
    struct const_iterator {
      friend class list<T>;
      //private:
        const node *ptr;
        
        const_iterator(const node *p): ptr(p) {}

      public:
        typedef T         value_type;
        typedef const T*  pointer;
        typedef const T&  reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        #ifdef USE_ITERATOR_TAGS
        typedef std::bidirectional_iterator_tag iterator_category;
        #endif

        const_iterator(const iterator &i): ptr(i.ptr) {}
        const_iterator(iterator &i): ptr(i.ptr) {}

        const_iterator& operator++() { // preincrement
          ptr = ptr->next;
          return *this;
        }

        const_iterator operator++(int) { // postincrement
          const node *tmp = ptr;
          ptr = ptr->next;
          return const_iterator(tmp);
        }

        const_iterator& operator--() { // predecrement
          ptr = ptr->prev;
          return *this;
        }

        const_iterator operator--(int) { // postdecrement
          node *tmp = ptr;
          ptr = ptr->prev;
          return const_iterator(tmp);
        }

        reference operator *() const { // dereference
          return ptr->data;
        }

        pointer operator ->() const {
          return &(ptr->data);
        }

        bool operator == (const const_iterator &i) const { return ptr == i.ptr; }
        bool operator != (const const_iterator &i) const { return ptr != i.ptr; }

    };

    /// Iterator used to iterate backwards through a list. 
    struct reverse_iterator { // deberia heredar de std::bidirectional_iterator<T, Distance>
      friend class list<T>;
      //private:
        iterator current;

        /// private constructor. we don't want reverse iterators created from iterators :)
        reverse_iterator(const iterator &i): current(i) {}

      public:
        typedef T         value_type;
        typedef T*        pointer;
        typedef T&        reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        #ifdef USE_ITERATOR_TAGS
        typedef std::bidirectional_iterator_tag iterator_category;
        #endif

        reverse_iterator(const reverse_iterator &other): current(other.current) {}

        reverse_iterator& operator++() { // preincrement
          --current;
          return *this;
        }

        reverse_iterator operator++(int) { // postincrement
          iterator tmp(current);
          --current;
          return reverse_iterator(tmp);
        }
        
        reverse_iterator& operator--() { // predecrement
          ++current;
          return *this;
        }

        reverse_iterator operator--(int) { // postdecrement
          iterator tmp(current);
          ++current;
          return reverse_iterator(tmp);
        }

        reference operator *() const { // dereference
          iterator tmp(current);
          return *(--tmp);
        }

        pointer operator ->() const {
          return &(operator*());
        }

        iterator base() {
          return current;
        }

        bool operator == (const reverse_iterator &other) const { return current == other.current; }
        bool operator != (const reverse_iterator &other) const { return current != other.current; }

    };

    /// Iterator used to iterate backwards through a list. 
    struct const_reverse_iterator { // deberia heredar de std::bidirectional_iterator<T, Distance>
      friend class list<T>;
      //private:
        const_iterator current;

        /// private constructor. we don't want reverse iterators created from iterators :)
        const_reverse_iterator(const const_iterator &ci): current(ci) {}

      public:
        typedef T         value_type;
        typedef const T*  pointer;
        typedef const T&  reference;
        typedef const T&  const_reference;
        typedef ptrdiff_t difference_type;
        #ifdef USE_ITERATOR_TAGS
        typedef std::bidirectional_iterator_tag iterator_category;
        #endif

        const_reverse_iterator(const const_reverse_iterator &other): current(other.current) {}

        const_reverse_iterator& operator++() { // preincrement
          --current;
          return *this;
        }

        const_reverse_iterator operator++(int) { // postincrement
          const_iterator tmp(current);
          --current;
          return const_reverse_iterator(tmp);
        }
        
        const_reverse_iterator& operator--() { // predecrement
          ++current;
          return *this;
        }

        const_reverse_iterator operator--(int) { // postdecrement
          const_iterator tmp(current);
          ++current;
          return const_reverse_iterator(tmp);
        }

        reference operator *() const { // dereference
          iterator tmp(current);
          return *(--tmp);
        }

        pointer operator ->() const {
          return &(operator*());
        }
        
        iterator base() {
          return current;
        }

        bool operator == (const reverse_iterator &other) const { return current == other.current; }
        bool operator != (const reverse_iterator &other) const { return current != other.current; }

    };

    /// Returns an iterator pointing to the beginning of the list. 
    iterator begin() { return iterator(sentinel.next); }

    /// Returns an iterator pointing to the end of the list. 
    iterator end()   { return iterator(&sentinel);  }

    /// Returns a const iterator pointing to the beginning of the list. 
    const_iterator begin() const { return const_iterator(sentinel.next); }

    /// Returns a const iterator pointing to the end of the list. 
    const_iterator end()   const { return const_iterator(&sentinel);  }
    
    /// Returns a reverse iterator pointing to the beginning of the reversed list. 
    reverse_iterator rbegin() { return reverse_iterator(end()); }

    /// Returns a reverse iterator pointing to the end of the reversed list. 
    reverse_iterator rend()   { return reverse_iterator(begin());  }

    /// Returns a const reverse iterator pointing to the beginning of the reversed list. 
    const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }

    /// Returns a const reverse iterator pointing to the end of the reversed list. 
    const_reverse_iterator rend()   const { return const_reverse_iterator(begin());  }


    /// Returns the size of the list. This is not required to be constant time
    /// by the standard. However, this is NOT std::list, and we have decided to
    /// keep size() constant time at the expense of having linear time splices.
    size_type  size()  const { return list_size;  }

    /// Returns the largest possible size of the list. 
    size_type  max_size() const { return size_type(-1); }

    /// true if the list's size is 0. 
    bool empty() const { return (list_size==0); }

    /// Creates an empty list. 
    list() { init_list(); }
    
    /// Creates a list with n elements, each of which is a copy of T(). 
    list(size_type n) {
      init_list();
      for (size_type i=0; i<n; i++)
        push_front(T());  
    }


    /// The copy constructor. 
    list(const list &l){
      init_list();
      for (const_iterator i = l.begin(); i != l.end(); i++)
        push_back(*i);
    }
    
    /// Creates a list with n copies of t. 
    list(size_type n, const_reference t) {
      init_fill(n, t);
    }

    private:
    template<typename I>
    void init_dispatch(I n, I x, list_aux_templates::_true_type) {
      init_fill(static_cast<size_type>(n), static_cast<const_reference>(x));  
    }

    template<typename InputIterator>
    void init_dispatch(InputIterator f, InputIterator l, list_aux_templates::_false_type) {
      init_range(f,l);
    }

    void init_fill(size_type n, const_reference t) {
      init_list();
      for (size_type i=0; i<n; i++)
        push_front(t);
    }

    /// Creates a list with a copy of a range.
    template <class InputIterator>
    void init_range(InputIterator f, InputIterator l) {
      init_list();
      for (; f != l; f++) {
        push_back(*f);
      }
    }
    
    public:
    template <class InputIterator>
    list(InputIterator f, InputIterator l) {
      // Check if it's an integer type. If it's integer, it's not an iterator
      typedef typename list_aux_templates::is_integer<InputIterator>::type IsInteger;
      init_dispatch(f,l, IsInteger());
    }

    /// The destructor. 
    ~list() {
      clear();
    }

    /// The assignment operator 
    list &operator=(const list &l){
      if (l != *this) {
	clear();
	for (iterator i = l.begin(); i != l.end(); i++)
	  push_back(*i);
      }
      return *this;
    }

    /// Returns the first element. 
    /// Precondition: !empty() 
    reference       front()       { return sentinel.next->data; }
    
    /// Returns the first element. 
    /// Precondition: !empty() 
    const_reference front() const { return sentinel.next->data; }
    
    /// Returns the last element. 
    /// Precondition: !empty() 
    reference       back()        { return sentinel.prev->data; }
    
    /// Returns the last element. 
    /// Precondition: !empty() 
    const_reference back() const  { return sentinel.prev->data; }

    /// Inserts a new element at the beginning. 
    void push_front(const_reference t) {
      insert(begin(), t);
    }

    /// Inserts a new element at the end. 
    void push_back(const_reference t)  {
      insert(end(), t);
    }

    /// Removes the first element. 
    /// Precondition: !empty() 
    void pop_front() {
      erase(begin());
    }

    /// Removes the last element. 
    /// Precondition: !empty() 
    void pop_back() {
      erase(iterator(sentinel.prev));
    }
    
    /// Swaps the contents of two lists. 
    void swap(list &l){
      node *first, *first_l;
      node *last,  *last_l;
      first   = sentinel.next;
      first_l = l.sentinel.next;
      last    = sentinel.prev;
      last_l  = l.sentinel.prev;
      AprilUtils::swap(first->prev, first_l->prev);
      AprilUtils::swap(last->next, last_l->next);
      AprilUtils::swap(sentinel.prev, l.sentinel.prev);
      AprilUtils::swap(sentinel.next, l.sentinel.next);
      AprilUtils::swap(list_size, l.list_size);
    }

    /// Inserts x before pos.
    /// Returns an iterator pointing to the inserted element
    iterator insert(iterator pos, const T& x) {
      node *ptr = pos.ptr;
      node *tmp = new node(x, ptr->prev, ptr);
      ptr->prev->next = tmp;
      ptr->prev = tmp;
      
      list_size++;
      return iterator(tmp);
    }
    
    private:
    template<typename I>
    void insert_dispatch(iterator pos, I n, I x, list_aux_templates::_true_type) {
      insert_copies(pos, static_cast<size_type>(n), static_cast<const_reference>(x));  
    }

    template<typename InputIterator>
    void insert_dispatch(iterator pos, InputIterator f, InputIterator l, list_aux_templates::_false_type) {
      insert_range(pos, f,l);
    }

    /// Inserts n copies of x before pos
    void insert_copies(iterator pos, size_type n, const_reference x) {
      for (size_type i = 0; i < n; i++)
        insert(pos, x);
    }

    /// Inserts the range [f, l) before pos
    template <class InputIterator>
    void insert_range(iterator pos, InputIterator f, InputIterator l) {
      for (InputIterator i = f; i != l; i++)
        insert(pos, *i);
    }
    
    public:

    /// Inserts the range [f, l) before pos. 
    template <class InputIterator>
    void insert(iterator pos, InputIterator f, InputIterator l) {
      typedef typename list_aux_templates::is_integer<InputIterator>::type IsInteger;
      insert_dispatch(pos, f, l, IsInteger());
    }

    /// Inserts n copies of x before pos. 
    void insert(iterator pos, size_type n, const T& x) {
      insert_copies(pos, n, x);
    }
 
    /// Erases the element at position pos. 
    /// Returns an iterator to the element immediately following the one that
    /// was erased.
    /// Precondition: pos must be a dereferenceable iterator
    iterator erase(iterator pos) {
      node *ptr = pos.ptr;
      node *next = ptr->next;
      ptr->prev->next = ptr->next;
      ptr->next->prev = ptr->prev;
      delete ptr;
      list_size--;
      return iterator(next);
    }

    /// Erases the elements in range [first, last)
    /// Returns last
    iterator erase(iterator first, iterator last) {
      while(first != last)
        first = erase(first);
      return last;
    }

    /// Erases all of the elements. 
    void clear() {
      erase(begin(), end());
    }

    /// Inserts or erases elements at the end such that the size becomes n.
    void resize(size_t n, T x=T()) {
      if (list_size < n) {
        while (list_size < n) {
          push_back(x);
        }
      } 
      else if (list_size > n) {
        while (list_size > n) {
          pop_back();
        }
      }

      list_size = n;
    }

    /// position must be a valid iterator in *this, and x must be a list that
    /// is distinct from *this. (That is, it is required that &x != this.) All of
    /// the elements of x are inserted before position and removed from x. All
    /// iterators remain valid, including iterators that point to elements of x.
    /// NOTE: This function is linear time in order to keep size() constant time
    void splice(iterator position, list<T>& x) {
      splice(position, x, x.begin(), x.end());
    }

    /// position must be a valid iterator in *this, and i must be a
    /// dereferenceable iterator in x. Splice moves the element pointed to by i
    /// from x to *this, inserting it before position. All iterators remain
    /// valid, including iterators that point to elements of x. If position
    /// == i or position == ++i, this function is a null operation. This function
    /// is linear time in order to keep size() constant time.
    void splice(iterator position, list<T>& x, iterator i) {
      iterator first,last;
      first = i;
      last  = ++i;
      splice(position, x, first, last);
    }

    /// position must be a valid iterator in *this, and [first, last)  must be
    /// a valid range in x. position may not be an iterator in the range [first,
    /// last). Splice moves the elements in [first, last) from x to *this,
    /// inserting them before position. All iterators remain valid, including
    /// iterators that point to elements of x. [3] This function is linear 
    /// time in order to keep size() constant time. 
    void splice(iterator position, list<T>& x, iterator f, iterator l) {
      node *pos   = position.ptr;
      node *first = f.ptr;
      node *last  = l.ptr->prev;

      // we're supposed to check if *this and x use the same allocators,
      // but this is not the STL and we don't have allocators, so x is not used
      // (iterators are sufficient)
      int count=0;
      for (iterator i = f; i != l; i++)
        ++count;

      x.list_size -= count;
      this->list_size += count;
      
      first->prev->next = last->next;
      last->next->prev = first->prev;

      first->prev = pos->prev;
      last->next  = pos;

      pos->prev->next = first;
      pos->prev = last;

    }


    /// Removes all elements that compare equal to val. The relative order of
    /// elements that are not removed is unchanged, and iterators to elements
    /// that are not removed remain valid. This function is linear time: it
    /// performs exactly size() comparisons for equality. 
    void remove(const T& val) {
      iterator i = begin();
      while(i != end()) {
        if (*i == val)
          i = erase(i);
        else
          i++;
      }
    }

    /// Removes all elements *i such that p(*i) is true. The relative order of
    /// elements that are not removed is unchanged, and iterators to elements
    /// that are not removed remain valid. This function is linear time: it
    /// performs exactly size() applications of p. 
    template<class Predicate> 
    void remove_if(Predicate p) {
      iterator i = begin();
      while(i != end()) {
        if (p(*i))
          i = erase(i);
        else
          i++;
      }
    }

    /// Removes all but the first element in every consecutive group of equal
    /// elements. The relative order of elements that are not removed is
    /// unchanged, and iterators to elements that are not removed remain valid.
    /// This function is linear time: it performs exactly size() - 1 comparisons
    /// for equality. 
    void unique(); 

    /// Removes all but the first element in every consecutive group of
    /// equivalent elements, where two elements *i and *j are considered
    /// equivalent if p(*i, *j) is true. The relative order of elements that are
    /// not removed is unchanged, and iterators to elements that are not removed
    /// remain valid. This function is linear time: it performs exactly size() -
    /// 1 comparisons for equality. 
    template<class BinaryPredicate>
    void unique(BinaryPredicate p);

    /// Both *this and x must be sorted according to operator<, and they must
    /// be distinct. (That is, it is required that &x != this.) This function
    /// removes all of x's elements and inserts them in order into *this. The
    /// merge is stable; that is, if an element from *this is equivalent to one
    /// from x, then the element from *this will precede the one from x. All
    /// iterators to elements in *this and x remain valid. This function is
    /// linear time: it performs at most size() + x.size() - 1 comparisons. 
    void merge(list<T>& x);

    /// Comp must be a comparison function that induces a strict weak ordering
    /// (as defined in the LessThan Comparable requirements) on objects of type
    /// T, and both *this and x must be sorted according to that ordering. The
    /// lists x and *this must be distinct. (That is, it is required that &x !=
    /// this.) This function removes all of x's elements and inserts them in
    /// order into *this. The merge is stable; that is, if an element from *this
    /// is equivalent to one from x, then the element from *this will precede the
    /// one from x. All iterators to elements in *this and x remain valid. This
    /// function is linear time: it performs at most size() + x.size() - 1
    /// applications of Comp. 
    template<class BinaryPredicate>
    void merge(list<T>& x, BinaryPredicate Comp);


    /// Reverses the order of elements in the list. All iterators remain valid
    /// and continue to point to the same elements. This function is linear
    /// time. 
    void reverse();

    /// Sorts *this according to operator<. The sort is stable, that is, the
    /// relative order of equivalent elements is preserved. All iterators remain
    /// valid and continue to point to the same elements. [6] The number of
    /// comparisons is approximately N log N, where N is the list's size.
    void sort();

    /// Comp must be a comparison function that induces a strict weak ordering
    /// (as defined in the LessThan Comparable requirements on objects of type T.
    /// This function sorts the list *this according to Comp. The sort is stable,
    /// that is, the relative order of equivalent elements is preserved. All
    /// iterators remain valid and continue to point to the same elements. [6]
    /// The number of comparisons is approximately N log N, where N is the list's
    /// size. 
    template<class BinaryPredicate>
    void sort(BinaryPredicate comp); 

  };

} // namespace AprilUtils

#endif

