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

#include <cfloat>
#include "disallow_class_methods.h"
#include "matrix.h"
#include "maxmin.h"
#include "MersenneTwister.h"
#include "min_heap.h"
#include "pair.h"
#include "point.h"
#include "qsort.h"
#include "referenced.h"

/// K-Nearest-Neighbors.
namespace KNN {

  /// KDTree class for KNN search. It is not a complete KDTree, it isn't allow
  /// to insert or remove points. All data is pushed as bi-dimensional matrices,
  /// and after that the KDTree is build.
  template<typename T>
  class KDTree : public Referenced {
    APRIL_DISALLOW_COPY_AND_ASSIGN(KDTree);
    
    typedef AprilUtils::vector< Point<T> > PointsList;
    static const size_t MEDIAN_APPROX_SIZE=40;
    
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
      APRIL_DISALLOW_ASSIGN(KDNode);
      
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
      const Point<T> &getPoint() const { return point; }
      const KDNode *getLeft() const { return left; }
      const KDNode *getRight() const { return right; }
      T getSplitValue() const { return split_value; }
    };
    
    /// Class base for KNN search. It implements the interface for searching
    /// decisions, stores the k-best (or 1-best), and indicates when to stop.
    class Searcher {
      APRIL_DISALLOW_COPY_AND_ASSIGN(Searcher);
      
    public:
      Searcher() { }
      virtual ~Searcher() { }
      virtual void process(const Point<T> &node_point) = 0;
      virtual int side(const T split_value, const int axis) const = 0;
      virtual bool leftIntersection(const T split_value, const int axis) const = 0;
      virtual bool rightIntersection(const T split_value, const int axis) const = 0;
    };
    
    /// Specialization of Searcher for the NN search
    class OneBestSearcher : public Searcher {
      APRIL_DISALLOW_COPY_AND_ASSIGN(OneBestSearcher);
      
      Point<T> &X;
      int best_id;
      double best_distance;
      const int D;
    public:
      OneBestSearcher(Point<T> &X, const int D) :
        Searcher(),
	X(X), best_id(-1), best_distance(DBL_MAX), D(D) {
      }
      virtual ~OneBestSearcher() { }
      virtual void process(const Point<T> &node_point) {
	const double current_distance = X.dist(node_point, D);
	if (current_distance < best_distance) {
	  best_distance = current_distance;
	  best_id       = node_point.getId();
	}
      }
      virtual int side(const T split_value, const int axis) const {
	if (X[axis] < split_value) return 0;
	else return 1;
      }
      virtual bool leftIntersection(const T split_value, const int axis) const {
	const double diff = X[axis] - split_value;
	const double intersection_distance = diff*diff;
	return !(intersection_distance > best_distance) || X[axis]<split_value;
      }
      virtual bool rightIntersection(const T split_value, const int axis) const {
	const double diff = X[axis] - split_value;
	const double intersection_distance = diff*diff;
	return !(intersection_distance > best_distance) || !(X[axis]<split_value);
      }
      int getOneBestIndex() const { return best_id; }
      double getOneBestDistance() const { return best_distance; }
    };
    
    /// Less functor for min_heap for KBestSearcher
    struct KbestPairLess {
      bool operator()(const AprilUtils::pair<int,double> &a,
		      const AprilUtils::pair<int,double> &b) {
	// we use > instead of < to convert the min_heap into a max_heap
	return a.second > b.second;
      }
    };
    
    /// Specialization of Searcher for the K-NN search. It uses a max_heap where
    /// the points are ordered by distance (desceding order), allowing to use
    /// the furthest point to define the hypersphere.
    class KBestSearcher : public Searcher {
      APRIL_DISALLOW_COPY_AND_ASSIGN(KBestSearcher);
      
      typedef AprilUtils::pair<int,double> HeapNode;
      typedef AprilUtils::min_heap<HeapNode,KbestPairLess> MaxHeapType;
      Point<T> &X;
      double kbest_distance;
      const int D, K;
      MaxHeapType max_heap;
    public:
      KBestSearcher(const int K, Point<T> &X, const int D) :
        Searcher(),
	X(X), kbest_distance(DBL_MAX), D(D), K(K), max_heap(K,KbestPairLess()) {
      }
      virtual ~KBestSearcher() { }
      virtual void process(const Point<T> &node_point) {
	const double current_distance = X.dist(node_point, D);
	if (current_distance < kbest_distance || max_heap.size() < K) {
	  HeapNode e(node_point.getId(), current_distance);
	  if (max_heap.size() == K) max_heap.pop();
	  max_heap.push(e);
	  const HeapNode &top = max_heap.top();
	  kbest_distance = top.second;
	}
      }
      virtual int side(const T split_value, const int axis) const {
	if (X[axis] < split_value) return 0;
	else return 1;
      }
      virtual bool leftIntersection(const T split_value, const int axis) const {
	const double diff = X[axis] - split_value;
	const double intersection_distance = diff*diff;
	return !(intersection_distance > kbest_distance) || X[axis]<split_value;
      }
      virtual bool rightIntersection(const T split_value, const int axis) const {
	const double diff = X[axis] - split_value;
	const double intersection_distance = diff*diff;
	return !(intersection_distance > kbest_distance) || !(X[axis]<split_value);
      }
      void getBestData(AprilUtils::vector<int> &indices,
		       AprilUtils::vector<double> &distances) {
	indices.resize(max_heap.size());
	distances.resize(max_heap.size());
	for (int i=max_heap.size()-1; i>=0; --i) {
	  april_assert(!max_heap.empty());
	  const HeapNode &top = max_heap.top();
	  indices[i]   = top.first;
	  distances[i] = top.second;
	  max_heap.pop();
	}
      }
    };
    
    // properties
    
    const int D; ///< The number of dimensions
    int N; ///< The number of points
    /// A vector with matrices which contains the data
    AprilUtils::vector< Basics::Matrix<T>* > matrix_vector;
    /// A vector with indices of first point index in matrix_vector
    AprilUtils::vector<int> first_index;
    /// The root node of the KDTree
    KDNode *root;
    Basics::MTRand *random; ///< A random number generator
    
    // for stats
    int number_of_processed_points;
    
    // private methods
    
    /// Builds a point given its index
    Point<T> makePoint(int index) {
      int row;
      Basics::Matrix<T> *m = getMatrixAndRow(index,row);
      return Point<T>(m, row, index);
    }
    
    /// Returns the position of the median given a list of point indices and the
    /// axis where to compare
    Point<T> computeMedian(const PointsList *points, const int axis) const {
      const size_t size = AprilUtils::min(MEDIAN_APPROX_SIZE, points->size());
      PointsList aux( static_cast<typename PointsList::size_type>(size) );
      for (size_t i=0; i<size; ++i) aux[i] = (*points)[random->randInt(size-1)];
      MedianCompare predicate(axis);
      return AprilUtils::Selection(aux.begin(), size, size/2, predicate);
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

    /// Generic search method, uses an instance of Searcher to compute
    /// intersections with hyperplane and the search hypersphere.
    void searchKNN(Searcher &searcher,
		   const KDNode *node,
		   const int depth) {
      if (node == 0) return;
      const Point<T> &node_point = node->getPoint();
      const T split_value = node->getSplitValue();
      const KDNode *left  = node->getLeft();
      const KDNode *right = node->getRight();
      const int axis      = depth%D;
      //
      ++number_of_processed_points;
      searcher.process(node_point);
      switch(searcher.side(split_value,axis)) {
      case 0:
	searchKNN(searcher, left, depth+1);
	if (searcher.rightIntersection(split_value,axis))
	  searchKNN(searcher, right, depth+1);
	break;
      case 1:
	searchKNN(searcher, right, depth+1);
	if (searcher.leftIntersection(split_value,axis))
	  searchKNN(searcher, left, depth+1);
	break;
      default:
	;
      }
    }

    /// For debugging purposes
    void print(const KDNode *node, int depth) {
      if (node == 0) return;
      for (int i=0; i<depth; ++i)
	printf("  ");
      printf("%d %f\n", node->getPoint().getId(), node->getSplitValue());
      print(node->getLeft(),  depth+1);
      print(node->getRight(), depth+1);
    }
    
  public:

    KDTree(const int D, Basics::MTRand *random) :
      D(D), N(0), root(0), random(random) {
      IncRef(random);
      first_index.push_back(0);
    }
    
    ~KDTree() {
      DecRef(random);
      for (typename AprilUtils::vector< Basics::Matrix<T>* >::iterator it=matrix_vector.begin();
	   it != matrix_vector.end(); ++it)
	DecRef(*it);
      delete root;
    }
    
    /// Returns a matrix and a row from an index point
    Basics::Matrix<T> *getMatrixAndRow(int index, int &row) {
      april_assert(index >= 0 && index < N);
      int izq,der,m;
      izq = 0; der = static_cast<int>(first_index.size());
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
    
    void pushMatrix(Basics::Matrix<T> *m) {
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
	Basics::Matrix<T> *m = matrix_vector[j];
	for (int row=0; row<m->getDimSize(0); ++row, ++i) {
	  april_assert(i<N);
	  (*points_list)[i] = Point<T>(m, row, i);
	}
      }
      root = build(points_list, 0);
      delete points_list;
    }
    
    /// Method for 1-NN search, it receives a matrix with one point, and returns
    /// the best point index (in the order of they were pushed), the distance to
    /// the best, and a matrix with the best point (if needed, that is, result
    /// pointer != 0).
    int searchNN(Basics::Matrix<T> *point_matrix,
		 double &distance,
		 Basics::Matrix<T> **result) {
      if (root == 0)
	ERROR_EXIT(256, "Build method needs to be called before searching\n");
      if (point_matrix->getNumDim() != 2 || point_matrix->getDimSize(0) != 1)
	ERROR_EXIT(256, "A bi-dimensional matrix with one row is needed\n");
      if (point_matrix->getDimSize(1) != D)
	ERROR_EXIT2(256, "Incorrect number of columns, expected %d, found %d\n",
		    D, point_matrix->getDimSize(1));
      Point<T> point(point_matrix, 0, -1);
      OneBestSearcher one_best(point,D);
      number_of_processed_points = 0;
      searchKNN(one_best, root, 0);
      int best_id = one_best.getOneBestIndex();
      distance = one_best.getOneBestDistance();
      if (result != 0) {
	int best_row;
	Basics::Matrix<T> *best_matrix = getMatrixAndRow(best_id, best_row);
	int coords[2] = { best_row, 0 };
	int sizes[2]  = { 1, D };
	*result = new Basics::Matrix<T>(best_matrix, coords, sizes, false);
      }
      return best_id;
    }

    /// Method for K-NN search (with K>1). Tt receives a matrix with one point
    /// and the K value, and returns a vector of indices, a vector of distances,
    /// and a vector of matrices (if needed, that is, result pointer != 0).
    void searchKNN(int K,
		   Basics::Matrix<T> *point_matrix,
		   AprilUtils::vector<int> &indices,
		   AprilUtils::vector<double> &distances,
		   AprilUtils::vector< Basics::Matrix<T> *> *result=0) {
      if (root == 0)
	ERROR_EXIT(256, "Build method needs to be called before searching\n");
      if (K == 1) {
	double distance;
	Basics::Matrix<T> *resultM;
	int best_id = searchNN(point_matrix,distance,(result!=0)?(&resultM):0);
	indices.push_back(best_id);
	distances.push_back(distance);
	if (result) result->push_back(resultM);
      }
      else {
	if (point_matrix->getNumDim() != 2 || point_matrix->getDimSize(0) != 1)
	  ERROR_EXIT(256, "A bi-dimensional matrix with one row is needed\n");
	if (point_matrix->getDimSize(1) != D)
	  ERROR_EXIT2(256, "Incorrect number of columns, expected %d, found %d\n",
		      D, point_matrix->getDimSize(1));
	Point<T> point(point_matrix, 0, -1);
	KBestSearcher k_best(K,point,D);
	number_of_processed_points = 0;
	searchKNN(k_best, root, 0);
	k_best.getBestData(indices,distances);
	if (result != 0) {
	  result->reserve(static_cast<size_t>(indices.size()));
	  for (size_t i=0; i<indices.size(); ++i) {
	    int best_id = indices[i];
	    int best_row;
	    Basics::Matrix<T> *best_matrix = getMatrixAndRow(best_id, best_row);
	    int coords[2] = { best_row, 0 };
	    int sizes[2]  = { 1, D };
	    result->push_back(new Basics::Matrix<T>(best_matrix, coords,
                                                    sizes, false));
	  }
	}
      }
    }
  
    int getDimSize() const { return D; }
    
    int size() const { return N; }
    
    /// For debugging purposes
    void print() {
      if (root == 0)
	ERROR_EXIT(256, "Build method needs to be called before print\n");
      print(root, 0);
    }
    
    int getNumProcessedPoints() const {
      return number_of_processed_points;
    }
  };
  typedef KDTree<float> KDTreeFloat;
}

#endif // KDTREE_H
