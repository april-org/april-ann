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
#ifndef MAXIPHOBIC_MINHEAP_H
#define MAXIPHOBIC_MINHEAP_H

// #include <algorithm> reemplazado por "swap.h"
#include "swap.h"
#include "vector.h"
#include <cstdlib>

using std::size_t;

namespace AprilUtils{

  template <typename T> class maxiphobic_minheap
  {
    struct node {
      int size;
      node *left;
      node *right;
      T data;
      
    node(const T& d): size(1), left(0), right(0), data(d) { }

    node(const T& d, node * const l, node * const r): 
      size(1), left(l), right(r), data(d) {
        if (l) size += l->size;
        if (r) size += r->size;
      }

#ifndef NO_POOL
      inline static void * operator new(size_t size){
        if (size != sizeof(node))
          return ::operator new(size);

        node *n = freelist;
        if (n != 0) {
          freelist = n->left;
          poolsize--;
          return n;
        } else {
          return ::operator new(size);
        }
      }
      
      inline static void operator delete(void *object, size_t size) {
        if (object == 0) return;

        if (size != sizeof(node)) {
          ::operator delete(object);
          return;
        }

        if (poolsize < MAXPOOLSIZE) {
          node *deleted_node = static_cast<node *>(object);
          deleted_node->left = freelist;
          freelist = deleted_node;
          poolsize++;
        } else {
          ::operator delete(object);
        }
      }
      private:
      static node *freelist;
      static int poolsize;
      static const int MAXPOOLSIZE=2000000;
#endif // NO_POOL
    };

    node *root;


    static node* merge(node *node1, node *node2) {

      if (node1 == 0) return node2;
      if (node2 == 0) return node1;
      node *cur=0;
      node *A, *B, *C;
      if (node1->data < node2->data){
        // merge(node1, node2) sobre node1
        cur = node1;
        node1->size += node2->size;
        C = node2;
      } else {
        // merge(node1, node2) sobre node2
        cur = node2;
        node2->size += node1->size;
        C = node1;
      }

      node *min = cur;

      A = cur->left;
      B = cur->right;

      while ((A != 0) && (B != 0) && (C != 0)){
        // Tenemos 3 arboles: A, B y C a partir de los cuales
        // hay que crear 2 hijos para cur.
        // Hacemos que A sea el de mayor tamaño
        if (B->size > A->size)
          swap(A, B);
        if (C->size > A->size)
          swap(A, C);

        cur->left = A;
        if (B->data < C->data) {
          // merge(B, C) sobre B
          cur->right = B;
          cur = B;
          B->size += C->size;
          A = B->left;
          B = B->right;
        } else {
          // merge(B, C) sobre C
          cur->right = C;
          cur = C;
          C->size += B->size;
          A = C->left;
          C = C->right;
        }
      }
      // Al salir del bucle al menos uno de A, B, C esta vacio.
      if (A==0) {
        cur->left = B;
        cur->right = C;
      } else if (B==0) {
        cur->left = A;
        cur->right = C;
      } else { // C==0
        cur->left = A;
        cur->right = B;
      }

      return min;
    }


    void delete_tree(node *n) {
      if (n->left  != 0) delete_tree(n->left);
      if (n->right != 0) delete_tree(n->right);
      delete n;
    }


    // Copia recursiva del arbol
    node *copy(node *other) {
      return (other == 0 ? 0 : new node(other->data, copy(other->left), copy(other->right)));
    }

    public:
    maxiphobic_minheap(): root(0) { }
    
    // Constructor de copia y operador de asignacion
    // para evitar que los generados por defecto nos produzcan un universo de dolor.
    // Si hacen falta se implementan :)
    maxiphobic_minheap(const maxiphobic_minheap &other) { root = copy(other.root); }
    maxiphobic_minheap& operator= (const maxiphobic_minheap& other) {
      // Primero borramos los elementos que ya tuvieramos
      if (root != 0) delete_tree(root);

      root = copy(other.root);
    }
    
    ~maxiphobic_minheap() {
      if (root != 0) delete_tree(root);
    }

    void clear() {
      delete_tree(root);
    }
    
    void insert(const T &data) {
      node *p = new node(data);
      root = merge(root, p);
    }

    void delete_min() {
      node *old_root = root;
      root = merge(root->left, root->right);
      delete old_root;
    }

    T read_min() {
      return root->data;
    }

    T extract_min() {
      T res = read_min();
      delete_min();
      return res;
    }

    void foreach(void (*func)(T)) {
      if (root) {
        // Usamos 2*ln(size) como estimacion de profundidad del arbol
        vector<node *> stack;
        node *cur;

        stack.reserve(2*int(log(root->size)));

        stack.push_back(root);
        while (!stack.empty()) {
          cur = stack.back();
          stack.pop_back();

          func(cur->data);

          if (cur->right) stack.push_back(cur->right);
          if (cur->left) stack.push_back(cur->left);
        }
      }
    }

    bool empty() const { return root == 0; }
    int size() const { return (empty()? 0 : root->size); }
  };

#ifndef NO_POOL
  template <typename T> int maxiphobic_minheap<T>::node::poolsize=0;
  template <typename T> typename maxiphobic_minheap<T>::node *maxiphobic_minheap<T>::node::freelist=0;
#endif // NO_POOL
}

#endif

