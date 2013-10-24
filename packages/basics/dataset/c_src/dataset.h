/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Francisco Zamora-Martinez, Jorge
 * Gorbe-Moya
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
#ifndef DATASET_H
#define DATASET_H

#include <cmath>
#include "unused_variable.h"
#include "referenced.h"
#include "matrix.h"
#include "referenced_vector.h"
#include <stdint.h>

using april_utils::ReferencedVectorUint;

/// A pure abstract templatized class that serves as interface.
/**
   A DataSet is a class interface which define the concept of set of pattern
   vectors. The template argument is the type of the data stored. Methods for
   get the size of the set, or size of each pattern vector, are defined. Also
   methods to put a new pattern or to get a pattern. Patterns are indexed from 0
   to N-1, being N the numPatterns of the DataSet.
*/
template <typename T>
class DataSet : public Referenced {
 public:
  virtual ~DataSet() { }
  /// Number of patterns in the set
  virtual int numPatterns()=0;
  /// Size of each pattern.
  virtual int patternSize()=0;
  /// Get the pattern index to the vector pat. The function call returns the
  /// patternSize().
  virtual int getPattern(int index, T *pat)=0;
  /// Put the given vector pat at pattern index. The function returns the
  /// patternSize().
  virtual int putPattern(int index, const T *pat)=0;
};

/// DataSet specialization to put or get patterns from a Matrix object.
template <typename T>
class MatrixDataSet : public DataSet<T> {
 private:
  /// A pointer referenced to the underlying Matrix object, shared pointer.
  Matrix<T> *matrix;
  /// Initial position of each dimension
  int *offset;
  /// subPattern size.
  int *subMatrixSize;
  /// Circular boolean indicator for each dimension.
  bool *circular;
  /// Default value of positions out of Matrix limits.
  T defaultValue;
  /// Step of each dimension for the sliding window.
  int *step;
  /// Number of movements on each dimension (number of steps).
  int *numSteps;
  /// Order of movement for each dimension.
  int *orderStep;
  void setValue(int *dest, int* orig);
  void setValue(bool *dest, bool* orig);
  int numPatternsv;
  int patternSizev;
  /// Auxiliar, for getPattern.
  int *coordinate;
  /// Auxiliar, for getPattern.
  T *pattern;
  /// Auxiliar, for putPattern.
  const T *const_pattern;
  /// Auxiliar, for getPattern.
  int offsetpat;
  void index2coordinate(int index);
  void auxGetPattern(int offsetmatrix, int d);
  void auxPutPattern(int offsetmatrix, int d);
 public:
  MatrixDataSet(Matrix<T> *m);
  virtual ~MatrixDataSet();
  /// Setter for offset attribute.
  void setOffset(int* v) {setValue(offset,v);}
  /// Setter for step attribute.
  void setStep(int* v) {setValue(step,v);}
  /// Setter for submatrixsize attribute.
  void setSubMatrixSize(int* v);
  /// Setter for numSteps attribute.
  void setNumSteps(int* v);
  /// Setter for circular attribute.
  void setCircular(bool* v) {setValue(circular,v);}
  /// Setter for orderStep attribute.
  void setOrderStep(int* v) {setValue(orderStep,v);}
  /// Setter for default value attribute.
  void setDefaultValue(T f){defaultValue=f;}
  
  int numPatterns() { return numPatternsv; }
  int patternSize() { return patternSizev; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// DataSet specialization to put or get patterns from a union of DataSets.
/**
   This object allows several number of DataSets (which has exactly the same
   patternSize) to be treated as an unique DataSet. The numPatterns is the
   addition of the numPatterns of each individual DataSet.
*/
template <typename T>
class UnionDataSet : public DataSet<T> {
 private:
  /// Number of DataSets.
  int num;
  /// Vector of referenced pointers to the DataSets.
  DataSet<T> **vds;
  /// Vector of num+1 indices.
  int *d;
  /// Pattern size.
  int patternsz;
 public:
  UnionDataSet(int n, DataSet<T>**v);
  virtual ~UnionDataSet();
  int numPatterns() { return d[num]; }
  int patternSize() { return patternsz; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// DataSet specialization which serves to getPatterns from an IdentityMatrix.
template <typename T>
class IdentityDataSet : public DataSet<T> {
 private:
  /// Value of zero.
  const T zerovalue,
  /// Value of one.
    onevalue;
  /// Patterm size.
  int patternsz;
 public:
  IdentityDataSet(int patternSize, 
		  T zerovalue,
		  T onevalue);
  virtual ~IdentityDataSet();
  int numPatterns() { return patternsz; }
  int patternSize() { return patternsz; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A DataSet specialization which takes an interval of patterns from other.
/**
   This object is specialized to take an interval of patterns from other
   DataSet. For more general subsets representation, see IndexDataSet
   documentation.
*/
template <typename T>
class SubDataSet : public DataSet<T> {
 private:
  /// Initial position of the interval
  int ini,
  /// Final position of the interval
    fin,
  /// Resulting numPatterns
    size;
  /// The underlying DataSet
  DataSet<T> *ds;
 public:
  SubDataSet(int ini, int fin, DataSet<T>*ds);
  virtual ~SubDataSet();
  int numPatterns() { return size; }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which takes an split of other DataSet.
/**
   This object allows to take a subset of the output elements from a given
   DataSet. That is, the resulting DataSet has the same numPatterns, but lower
   patternSize. The output subset is an interval.
 */
template <typename T>
class SplitDataSet : public DataSet<T> {
 private:
  /// Initial position of the split interval.
  int ini,
  /// Final position of the split interval.
    fin,
  /// Resulting patternSize.
    size;
  /// An auxiliary vector.
  T *aux;
  /// The underlying DataSet
  DataSet<T> *ds;
 public:
  SplitDataSet(int ini, int fin, DataSet<T> *ds);
  virtual ~SplitDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return size; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat) {
    UNUSED_VARIABLE(index);
    UNUSED_VARIABLE(pat);
    return 0;
  } // NO DEFINIDO
};

/// A specialization of DataSet which join outputs from several DataSets.
/**
   This object joins the output a several number of given DataSets. All DataSets
   must have exactly the same numPatterns. The resulting patternSize is the
   addition of each patternSize of individuals DataSets.
 */
template <typename T>
class JoinDataSet : public DataSet<T> {
 private:
  /// Number of DataSets.
  int num;
  /// A vector of referenced pointers to the underlying DataSets.
  DataSet<T> **vds;
  /// num+1 size vector which keep patternSize accumulated result.
  int *d;
 public:
  JoinDataSet(int n, DataSet<T> **v);
  virtual ~JoinDataSet();
  int numPatterns() { return vds[0]->numPatterns(); }
  int patternSize() { return d[num]; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which takes a general sub set from other.
/**
   This object allows to take any subset of patterns from other DataSet. It
   builds a mapping between indexes and patterns. It is also useful for
   classification output specification.
 */
template <typename T>
class IndexDataSet : public DataSet<T> {
  // A float could represent the int value 16777216
 private:
  /// DataSet from which take indexes.
  DataSet<T> *indices;
  /// Number of dictionaries, we need indices->patternSize() dictionaries.
  int numdiccionarios;
  /// Dictionary DataSets from which take patterns.
  DataSet<T> **diccionarios;
  /// Auxiliar vector.
  T *patternindices;
  /// Output pattern size.
  int patternsize;
  /// First index, to solve Lua/C problems because of Lua 1 first index.
  int firstindex;
 public:
  IndexDataSet(DataSet<T> **datasets,int firstindex=0);
  virtual ~IndexDataSet();
  int numPatterns() { return indices->numPatterns(); }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// An auxiliar object for LinearCombDataSet.
template <typename T>
class LinearCombConf : public Referenced {
  public:
  /// Pattern size of the linear combination
  int patternsize;
  /// Number of pairs (index,weight) of each patternSize.
  int *numTuplas;
  /// Indices of the combinations.
  int *indices;
  /// Weights of the combinations.
  T *pesos;

  LinearCombConf(int pattsz, int numpairs) {
    patternsize = pattsz;
    numTuplas = (new int[pattsz+1])+1;
    numTuplas[-1] = 0;
    indices = new int[numpairs];
    pesos = new T[numpairs];
  }
  virtual ~LinearCombConf() { delete[] (numTuplas-1); delete[] indices; delete[] pesos; }
};

/// A specialization of DataSet, takes a DataSet and outputs a linear combination of its patterns.
/**
   This object takes a DataSet and produces a linear combination of its
   patterns. For instance, taking a DataSet with X patternSize and generates Y
   patternSize patterns, the object will need a matrix of XxY. However, this
   matrix will be normally very disperse, so, the object LinearCombConf stores
   pairs of (index,weight), and has a list of pairs for each Y output element.
 */
template <typename T>
class LinearCombDataSet : public DataSet<T> {
 private:
  /// Linear combination configuration.
  LinearCombConf<T> *conf;
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// Auxiliar vector.
  T *aux;
 public:
  LinearCombDataSet(DataSet<T> *ds,
                    LinearCombConf<T> *conf);
  virtual ~LinearCombDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return conf->patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat) {
    UNUSED_VARIABLE(index);
    UNUSED_VARIABLE(pat);
    return 0;
  }
};

/// A specialization of DataSet which takes a DataSet and adds context to its patterns.
/**
   This object adds context to the patterns of a given DataSet. The context are
   adjacents patterns, at left or/and at right positions from the interesting
   pattern index. In the case of a border limit, the interesting pattern will be
   repeated.
 */
template <typename T>
class ContextualizerDataSet : public DataSet<T> {
 private:
  /// Left context.
  int ctxtizq,
  /// Right context.
    ctxtder;
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// Number of patterns.
  int numpatterns,
  /// Size of each pattern.
    patternsize;
  /// Indicates if the context will be reversed.
  bool reverse;
 public:
  ContextualizerDataSet(DataSet<T> *ds,int izq, int der, bool reverse=false);
  virtual ~ContextualizerDataSet();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet for data accumulation using putPattern
template <typename T>
class AccumulateDataSet : public DataSet<T> {
 private:
  /// Number of patterns
  int numpatterns,
  /// Pattern size
    patternsize;
  /// Data accumulation vector.
  double *data; // IMPORTANT: it is not T type, T will be casted from/to double
 public:
  AccumulateDataSet(int patsz, int numpat);
  virtual ~AccumulateDataSet();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

template <typename T>
class ByteDataSet : public DataSet<T> {
  // sirve para acumular una serie de valores mediante putpattern
 private:
  int numpatterns,patternsize;
  double a,b; // y = a*x+b con x entre 0 y 255
  unsigned char *data;
 public:
  ByteDataSet(int patsz, int numpat, double a, double b);
  virtual ~ByteDataSet();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

template <typename T>
class BitDataSet : public DataSet<T> {
 private:
  int numpatterns, patternsize;
  unsigned char *data;
 public:
  BitDataSet(int nump, int patsize);
  virtual ~BitDataSet();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};


/// A specialization of DataSet for use sparse data representation.
/**
   This object takes a matrix like this:
   
   \verbatim
   5 2 1 13 2 14 3 15 4 16 5
   3 20 1 13 2 21 3
   2 10 1 33 2
   ...
   \endverbatim
   
   where first N number indicates how many elements, and following N*2 numbers
   are pairs of (index,value).
 */
template <typename T>
class SparseDataset : public DataSet<T> {
  /// The underlying matrix with sparse data representation.
  Matrix<T> *matrix;
  /// Number of patterns.
  int numpatterns,
  /// Pattern size.
    patternsize;
  /// Index position at the given matrix for each pattern.
  int *matrix_indexes;
  
 public:
  SparseDataset(Matrix<T> *m, int nump, int patsize);
  virtual ~SparseDataset();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization DataSet for shortlist approach in NNLMs.
/**
   This object takes a DataSet as a dictionary of word features, and each
   getPattern return features of given word index. If the index is greater than
   short_list_size, it will take unk_word index.
 */
template <typename T>
class ShortListDataSet : public DataSet<T> {
  /// The underlying DataSet
  DataSet<T>	*ds;
  /// Size of the shortlist approach.
  int            short_list_size,
  /// Index of the unknown word (out-of-shortlist word)
    unk_word,
  /// Pattern size, number of features for each word
    patsize;
  
public:
  ShortListDataSet(DataSet<T> *ds, int short_list_size, int unk_word);
  virtual ~ShortListDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return patsize; }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// Similar to IndexedDataSet but using a vector of unsigned int as indexes.
template <typename T>
class IndexFilterDataSet : public DataSet<T> {
  /// The indexed DataSet.
  DataSet<T> *ds;
  /// The vector of indexes.
  ReferencedVectorUint *indexes;
 public:
  IndexFilterDataSet(DataSet<T> *ds, ReferencedVectorUint *indexes);
  virtual ~IndexFilterDataSet();
  int numPatterns() { return (int)indexes->size(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

#include "MersenneTwister.h"
/// A specialization of DataSet which add gaussian noise to patterns.
template <typename T>
class PerturbationDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// A MTRand for random gaussian noise generation
  MTRand     *random;
  /// Mean of the gaussian noise
  double      mean,
  /// Variance of the gaussian noise
    variance;
 public:
  PerturbationDataSet(DataSet<T> *ds, MTRand *random,
		      double mean, double variance);
  virtual ~PerturbationDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which add a fixed size salt noise to patterns.
template <typename T>
class SaltNoiseDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// A MTRand for random selection of pattern components
  MTRand     *random;
  /// Percentage of values to be modified
  double      vd;
  /// Value of the zero
  const T     zero;
  /// Number of zeroes: vd * patternSize
  int         number_of_zeroes;
  /// Vector of ints for the selected zero positions
  int *zero_positions;
public:
  SaltNoiseDataSet(DataSet<T> *ds, MTRand *random,
		   double vd, T zero);
  virtual ~SaltNoiseDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which add a fixed size salt and pepper noise to patterns.
template <typename T>
class SaltPepperNoiseDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// A MTRand for random selection of pattern components
  MTRand     *random;
  /// Percentage of values to be modified
  double      vd;
  /// Value of the zero
  const T     zero;
  /// Value of the one
  const T     one;
  /// Number of zeroes: vd * patternSize
  int         number_of_perturbations;
  /// Vector of ints for the selected zero positions
  int *perturbed_positions;
public:
  SaltPepperNoiseDataSet(DataSet<T> *ds, MTRand *random,
		   double vd, T zero, T one);
  virtual ~SaltPepperNoiseDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which takes samples on a defined step size.
template <typename T>
class StepDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// A MTRand for random selection of pattern components
  int step;
public:
  StepDataSet(DataSet<T> *ds, int step);
  virtual ~StepDataSet();
  int numPatterns() { 
      return (int)ceil((float)ds->numPatterns()/step); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};
/// A specialization of DataSet for cacheNNLMs training.
template <typename T>
class CacheDataSet : public DataSet<T> {
  
  DataSet<T>	*ds;
  int	       **word2cache;
  int		*word2cache_sizes;
  T		*decays;
  int            voc_size;
  int		 cache_size;
  T		 near_zero;
  int            max_history;
  int		 begin_token_id;
  int		 end_token_id;
  int		 null_token_id;
  int		 cache_stop_token_id;

 public:
  CacheDataSet(DataSet<T>	*ds, int **word2cache,
	       int		*word2cache_sizes, T *decays,
	       int		 voc_size,
	       int		 cache_size, T near_zero,
	       int		 begin_token_id, int end_token_id,
	       int		 null_token_id, int cache_stop_token_id);
  
  virtual ~CacheDataSet();
  int				 numPatterns() { return ds->numPatterns(); }
  int patternSize() { return cache_size; }
  int				 getPattern(int index, T *pat);
  int				 putPattern(int index, const T *pat);
};

template <typename T>
class DerivDataSet : public DataSet<T> {
  DataSet<T> *ds;
  bool deriv0,deriv1,deriv2;
  int numpatterns,origpatternsz,patternsz;
  T *orig,*left1,*left2,*right1,*right2;
 public:
  DerivDataSet(DataSet<T> *ds, bool deriv0, bool deriv1, bool deriv2);
  virtual ~DerivDataSet();
  int numPatterns() { return numpatterns; }
  int patternSize() { return patternsz;   }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which performs sub, and div normalization
template <typename T>
class SubAndDivNormalizationDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  /// The vector with sub values
  T          *sub;
  /// The vector with div values
  T          *div;
 public:
  SubAndDivNormalizationDataSet(DataSet<T> *ds, T *sub, T *div);
  virtual ~SubAndDivNormalizationDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/// A specialization of DataSet which performs a clamp
template <typename T>
class ClampDataSet : public DataSet<T> {
  /// The underlying DataSet.
  DataSet<T> *ds;
  float lower, upper;
 public:
  ClampDataSet(DataSet<T> *ds, float lower, float upper);
  virtual ~ClampDataSet();
  int numPatterns() { return ds->numPatterns(); }
  int patternSize() { return ds->patternSize(); }
  int getPattern(int index, T *pat);
  int putPattern(int index, const T *pat);
};

/*** Implementacion ***/
#include "dataset.cc"

#endif // DATASET_H
