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
#ifndef KDTREE_H
#define KDTREE_H

#include "MersenneTwister.h"
#include "referenced.h"
#include "point.h"
#include "qsort.h"
#include "matrix.h"
#include "maxmin.h"

namespace KNN {

  /// KDTree class, for fast KNN
  template<typename T>
  class KDTree : public Referenced {
    typedef april_utils::vector< Point<T> > PointsList;
    static const size_t MEDIAN_APPROX_SIZE=20;

    // For median computation
    struct MedianCompare {
      const int axis;
      MedianCompare(int axis) : axis(axis) { }
      bool operator()(const Point<T> &a, const Point<T> &b) {
	return a[axis] < b[axis];
      }
    };
    
    // Node of the KDTree
    class KDNode {
      friend class KDTree;
      Point<T>    point;
      KDNode     *left;
      KDNode     *right;
      
      const T split_value; ///< Used to split the hyperspace in two.
      
    public:
      KDNode(Point<T> &point, KDNode *left, KDNode *right, T split_value) :
	point(point), left(left), right(right), split_value(split_value) {
      }
      ~KDNode() {
	delete left;
	delete right;
      }
      
    };

    // properties
    
    const int D; ///< The number of dimensions
    int N; ///< The number of points
    /// A vector with matrices which contains the data
    april_utils::vector< Matrix<T>* > matrix_vector;
    /// A vector with indices of first point index in matrix_vector
    april_utils::vector<int> first_index;
    /// The root node of the KDTree
    KDNode *root;
    MTRand *random; ///< A random number generator
    
    // private methods
    
    /// Builds a point given its index
    Point<T> makePoint(int index) {
      int row;
      Matrix<T> *m = getMatrixAndRow(index,row);
      return Point<T>(m, row, index);
    }
    
    /// Returns the position of the median given a list of point indices and the
    /// axis where to compare
    Point<T> computeMedian(const PointsList *points, const int axis) const {
      const size_t size = april_utils::min(MEDIAN_APPROX_SIZE, points->size());
      PointsList aux( static_cast<typename PointsList::size_type>(size) );
      for (size_t i=0; i<size; ++i) aux[i] = (*points)[random->randInt(size-1)];
      MedianCompare predicate(axis);
      return april_utils::Selection(aux.begin(), size, size/2, predicate);
    }
    
    /// Splits into two PointsList using the given pivot (excluding it)
    void split(const PointsList *points_list, const Point<T> which, const int axis,
	       PointsList *left, PointsList *right) {
      T value = which[axis];
      for (const Point<T> *it = points_list->begin();
	   it != points_list->end(); ++it) {
	if ( (*it) != which) {
	  if ( (*it)[axis] < value ) left->push_back( *it );
	  else right->push_back(*it);
	}
      }
    }
    
    /// Builds recursively the KDTree, returning the node which contains all the
    /// underlying points
    KDNode *build(const PointsList *points_list, const int depth) {
      if (points_list == 0 || points_list->size() == 0) return 0;
      int axis      = depth%D;
      Point<T> median = computeMedian(points_list, axis);
      PointsList *left_split, *right_split;
      left_split  = new PointsList();
      right_split = new PointsList();
      left_split->reserve(points_list->size()/2);
      right_split->reserve(points_list->size()/2);
      split(points_list, median, axis, left_split, right_split);
      //
      KDNode *node =  new KDNode( median,
				  build(left_split,  depth+1),
				  build(right_split, depth+1),
				  median[axis]);
      //
      delete left_split;
      delete right_split;
      return node;
    }

    /// For debugging purposes
    void print(KDNode *node, int depth) {
      if (node == 0) return;
      for (int i=0; i<depth; ++i)
	printf("  ");
      printf("%d %f\n", node->point.getId(), node->split_value);
      print(node->left,  depth+1);
      print(node->right, depth+1);
    }
    
  public:

    KDTree(const int D, MTRand *random) : D(D), N(0), root(0), random(random) {
      IncRef(random);
      first_index.push_back(0);
    }
    
    ~KDTree() {
      DecRef(random);
      for (typename april_utils::vector< Matrix<T>* >::iterator it=matrix_vector.begin();
	   it != matrix_vector.end(); ++it)
	DecRef(*it);
    }
    
    /// Returns a matrix and a row from an index point
    Matrix<T> *getMatrixAndRow(int index, int &row) {
      int izq,der,m;
      izq = 0; der = N;
      do {
	m = (izq+der)/2;
	if (first_index[m] <= index) 
	  izq = m; 
	else 
	  der = m;
      } while (izq < der-1);
      row = index - first_index[izq];
      return matrix_vector[izq];
    }
    
    void pushMatrix(Matrix<T> *m) {
      if (m->getNumDim() != 2)
	ERROR_EXIT(256,
		   "Incorrect number of dimensions, expected bi-dimensional\n");
      if (m->getDimSize(1) != D)
	ERROR_EXIT2(256, "Incorrect number of columns, expected %d, found %d\n",
		    D, m->getDimSize(1));
      // accumulate the number of rows in order to compute the total number of
      // points
      N += m->getDimSize(0);
      matrix_vector.push_back(m);
      IncRef(m);
      //
      first_index.push_back(N);
    }
    
    /// Builds the KDTree with all the pushed matrix data. Previous computation
    /// will be deleted if exists.
    void build() {
      delete root;
      PointsList *points_list = new PointsList(N);
      int i=0;
      for (size_t j=0; j<matrix_vector.size(); ++j) {
	Matrix<T> *m = matrix_vector[j];
	for (int row=0; row<m->getDimSize(0); ++row, ++i) {
	  april_assert(i<N);
	  (*points_list)[i] = Point<T>(m, row, i);
	}
      }
      root = build(points_list, 0);
      delete points_list;
    }
    
    void searchNN(Matrix<T> *point) {
      
    }
    
    /// For debugging purposes
    void print() {
      print(root, 0);
    }
    
  };
  typedef KDTree<float> KDTreeFloat;
}

#endif // KDTREE_H
