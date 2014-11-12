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
#ifndef UTILDATASETTOKEN_H
#define UTILDATASETTOKEN_H

#include "dataset.h"
#include "datasetFloat.h"
#include "function_interface.h"
#include "matrixFloat.h"
#include "matrix_operations.h"
#include "smart_ptr.h"
#include "token_base.h"
#include "token_matrix.h"
#include "token_vector.h"
#include "table_of_token_codes.h"
#include "unused_variable.h"
#include "vector.h"

namespace Basics {

  class DataSetToken : public Referenced {
  public:
    virtual ~DataSetToken() { }
    /// Number of patterns in the set
    virtual int numPatterns()=0;
    /// Size of each pattern.
    virtual int patternSize()=0;
    /// Get the pattern index to the vector pat
    virtual Token *getPattern(int index)=0;
    /// Get the pattern index to the vector pat
    virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size) {
      Token *output;
      Token *aux_token = getPattern(indexes[0]);
      TokenCode token_code = aux_token->getTokenCode();
      switch(token_code) {
      case table_of_token_codes::token_matrix: {
        // FIXME: Check this shape to be consistent when getPattern() returns a multi-dimensional matrix.
        int dims[2]   = { static_cast<int>(bunch_size),
                          patternSize() };
        MatrixFloat *output_mat = new MatrixFloat(2, dims);
        IncRef(output_mat);
        TokenMatrixFloat *output_mat_token  = new TokenMatrixFloat(output_mat);
        output           = output_mat_token;
        // the dims vector will be used to rewrap the matrices to has the same
        // structure
        dims[0] = patternSize();
        MatrixFloat *submat = 0;
        for (int i=0; i<static_cast<int>(bunch_size); ++i) {
          if (i>0) aux_token = getPattern(indexes[i]);
          IncRef(aux_token);
	
          TokenMatrixFloat *aux_mat_token;
          aux_mat_token = aux_token->convertTo<TokenMatrixFloat*>();
          MatrixFloat *aux_mat = aux_mat_token->getMatrix();
          MatrixFloat *aux_mat_rewrapped = aux_mat->rewrap(dims, 1);
          IncRef(aux_mat);
          IncRef(aux_mat_rewrapped);
	
          submat = output_mat->select(0, i, submat);
          AprilMath::MatrixExt::Operations::matCopy(submat,aux_mat_rewrapped);
	
          DecRef(aux_mat_rewrapped);
          DecRef(aux_mat);
          DecRef(aux_token);
        }
        delete submat;
        DecRef(output_mat);
        break;
      }
      default: {
        TokenBunchVector *output_token_vector = new TokenBunchVector(bunch_size);
        output         = output_token_vector;
        unsigned int i = 0;
        do {
          (*output_token_vector)[i] = aux_token;
          IncRef(aux_token);
          if ( (++i) < bunch_size) aux_token = getPattern(indexes[i]);
          else break;
        } while(true);
        break;
      }
      }
      return output;
    }
    /// Put the given vector pat at pattern index
    virtual void putPattern(int index, Token *pat)=0;
    /// Put the pattern bunch
    virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
                                 Token *pat)=0;
  };

  class DataSetTokenVector : public DataSetToken {
    AprilUtils::vector<Token*> data;
    int pattern_size;
  public:
    DataSetTokenVector(int pattern_size) : pattern_size(pattern_size) { }
    virtual ~DataSetTokenVector() {
      for (unsigned int i=0; i<data.size(); ++i) DecRef(data[i]);
    }
    void push_back(Token *token) {
      data.push_back(token);
      IncRef(token);
    }
    /// Number of patterns in the set
    virtual int numPatterns() { return static_cast<int>(data.size()); }
    /// Size of each pattern.
    virtual int patternSize() { return pattern_size; }
    /// Get the pattern index to the vector pat
    virtual Token *getPattern(int index) {
      if (index < 0 || index >= numPatterns())
        return 0;
      return data[index]->clone();
    }
    /// Put the given vector pat at pattern index
    virtual void putPattern(int index, Token *pat) {
      if (index < 0 || index >= numPatterns()) return;
      AssignRef(data[index], pat);
    }
    /// Put the pattern bunch
    virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
                                 Token *pat) {
      UNUSED_VARIABLE(indexes);
      UNUSED_VARIABLE(bunch_size);
      UNUSED_VARIABLE(pat);
      ERROR_EXIT(128, "Not implemented!!!\n");    
    }
  };

  class UnionDataSetToken : public DataSetToken {
    AprilUtils::vector<DataSetToken*> ds_array;
    AprilUtils::vector<int>           sum_patterns;
    int pattern_size;

  public:
    UnionDataSetToken(DataSetToken **ds_array, unsigned int size) :
      pattern_size(0) {
      sum_patterns.push_back(0);
      for (unsigned int i=0; i<size; ++i) push_back(ds_array[i]);
    }
    UnionDataSetToken() : pattern_size(0) {
      sum_patterns.push_back(0);
    }
    virtual ~UnionDataSetToken() {
      for (unsigned int i=0; i<ds_array.size(); ++i) DecRef(ds_array[i]);
    }
    void push_back(DataSetToken *ds) {
      ds_array.push_back(ds);
      IncRef(ds);
      if (pattern_size == 0)
        pattern_size = ds->patternSize();
      else if (pattern_size != ds->patternSize())
        ERROR_EXIT2(128, "Incorrect pattern size, expected %d, found %d\n",
                    pattern_size, ds->patternSize());
      sum_patterns.push_back(sum_patterns.back() + ds->numPatterns());
    }
    /// Number of patterns in the set
    virtual int numPatterns() { return sum_patterns.back(); }
    /// Size of each pattern.
    virtual int patternSize() { return pattern_size; }
    /// Get the pattern index to the vector pat
    virtual Token *getPattern(int index) {
      if (index < 0 || index >= numPatterns())
        return 0;
      int izq,der,m;
      izq = 0; der = static_cast<int>(sum_patterns.size());
      do {
        m = (izq+der)/2;
        if (sum_patterns[m] <= index) 
          izq = m; 
        else 
          der = m;
      } while (izq < der-1);
      return ds_array[izq]->getPattern(index-sum_patterns[izq]);
    }
    /// Put the given vector pat at pattern index
    virtual void putPattern(int index, Token *pat) {
      UNUSED_VARIABLE(index);
      UNUSED_VARIABLE(pat);
      ERROR_EXIT(128, "Not implemented!!!\n");
    }
    /// Put the pattern bunch
    virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
                                 Token *pat) {
      UNUSED_VARIABLE(indexes);
      UNUSED_VARIABLE(bunch_size);
      UNUSED_VARIABLE(pat);
      ERROR_EXIT(128, "Not implemented!!!\n");
    }
  };

  class DataSetFloat2TokenWrapper : public DataSetToken {
    AprilUtils::SharedPtr<MatrixFloat> aux_mat;
    AprilUtils::SharedPtr<DataSetFloat> ds;
  public:
    DataSetFloat2TokenWrapper(DataSetFloat *ds) :
      ds(ds) {
      int dims[2] = { 1, ds->patternSize() };
      aux_mat = new MatrixFloat(2, dims);
#ifdef USE_CUDA
      aux_mat->setUseCuda(false);
#endif
    }
    virtual ~DataSetFloat2TokenWrapper() { }
    int numPatterns() { return ds->numPatterns(); }
    int patternSize() { return ds->patternSize(); }
    Token *getPattern(int index) {
      int dims[2] = { 1, patternSize() };
      MatrixFloat *mat = new MatrixFloat(2, dims);
      TokenMatrixFloat *token = new TokenMatrixFloat(mat);
      AprilMath::FloatGPUMirroredMemoryBlock *mem_block = mat->getRawDataAccess();
      float *mem_ptr = mem_block->getPPALForWrite();
      ds->getPattern(index, mem_ptr);
      return token;
    }
    Token *getPatternBunch(const int *indexes, unsigned int bunch_size) {
      int dims[2], major_dim=0;
      dims[0] = static_cast<int>(bunch_size); dims[1] = patternSize();
      MatrixFloat *mat = new MatrixFloat(2, dims);
#ifdef USE_CUDA
      bool old_use_cuda = mat->getCudaFlag();
      mat->setUseCuda(false);
#endif
      // The TokenMatrixFloat takes increases reference counter of Matrix.
      TokenMatrixFloat *token = new TokenMatrixFloat(mat);
      // The memory block given to ds->getPattern(...).
      AprilMath::FloatGPUMirroredMemoryBlock *aux_mem_block = aux_mat->getRawDataAccess();
      float *aux_mem = aux_mem_block->getPPALForWrite();
      // int pattern_size = patternSize();
      int num_patterns = numPatterns();
      dims[major_dim] = 1;
      AprilUtils::SharedPtr<MatrixFloat> submat;
      for (unsigned int i=0; i<bunch_size; ++i) {
        april_assert(0 <= indexes[i] && indexes[i] < num_patterns);
        ds->getPattern(indexes[i], aux_mem);
        submat = mat->select(0, i, submat.get());
        AprilMath::MatrixExt::Operations::matCopy(submat.get(),aux_mat.get());
      }
#ifdef USE_CUDA
      mat->setUseCuda(old_use_cuda);
#endif
      return token;
    }
    void putPattern(int index, Token *pat) {
      if (pat->getTokenCode() != table_of_token_codes::token_matrix)
        ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
      TokenMatrixFloat *token_matrix = pat->convertTo<TokenMatrixFloat*>();
      MatrixFloat *mat = token_matrix->getMatrix();
      AprilMath::FloatGPUMirroredMemoryBlock *mem_block = mat->getRawDataAccess();
      const float *v = mem_block->getPPALForRead();
      ds->putPattern(index, v);
    }
  
    void putPatternBunch(const int *indexes,unsigned int bunch_size,
                         Token *pat) {
      if (pat->getTokenCode() != table_of_token_codes::token_matrix)
        ERROR_EXIT(128, "Incorrect token type, expected token matrix\n");
      TokenMatrixFloat *token_matrix = pat->convertTo<TokenMatrixFloat*>();
      MatrixFloat *mat = token_matrix->getMatrix();
#ifdef USE_CUDA
      bool old_use_cuda = mat->getCudaFlag();
      mat->setUseCuda(false);
#endif
      if (mat->getNumDim() != 2)
        ERROR_EXIT(128, "Only allowed for 2-dim matrices\n");
      AprilMath::FloatGPUMirroredMemoryBlock *aux_mem_block = aux_mat->getRawDataAccess();
      float *aux_mem = aux_mem_block->getPPALForWrite();
      int pattern_size = patternSize();
      int coords[2], major_dim;
      int sizes[2];
      major_dim = 0;
      coords[1] = 0;
      sizes[0]  = 1;
      sizes[1]  = pattern_size;
      AprilUtils::SharedPtr<MatrixFloat> submat;
      for (unsigned int i=0; i<bunch_size; ++i) {
        coords[major_dim] = static_cast<int>(i);
        submat = mat->select(0, i, submat.get());
        AprilMath::MatrixExt::Operations::matCopy(aux_mat.get(),submat.get());
        ds->putPattern(indexes[i], aux_mem);
      }
#ifdef USE_CUDA
      mat->setUseCuda(old_use_cuda);
#endif
    }
  };

  class DataSetTokenFilter : public DataSetToken {
    // the underlying dataset
    DataSetToken *ds;
    Functions::FunctionInterface *filter;
  public:
    DataSetTokenFilter(DataSetToken *ds,
                       Functions::FunctionInterface *filter) :
      ds(ds), filter(filter) {
      IncRef(ds);
      IncRef(filter);
    }
    virtual ~DataSetTokenFilter() {
      DecRef(ds);
      DecRef(filter);
    }
    /// Number of patterns in the set
    virtual int numPatterns() { return ds->numPatterns(); }
    /// Size of each pattern.
    virtual int patternSize() {
      return ( (filter->getOutputSize() > 0) ?
               filter->getOutputSize() :
               ds->patternSize() );
    }
    /// Get the pattern index to the vector pat
    virtual Token *getPattern(int index) {
      return filter->calculate(ds->getPattern(index));
    }
    /// Get the pattern index to the vector pat
    virtual Token *getPatternBunch(const int *indexes,unsigned int bunch_size) {
      return filter->calculate(ds->getPatternBunch(indexes,bunch_size));
    }
    /// Put the given vector pat at pattern index
    virtual void putPattern(int index, Token *pat) {
      UNUSED_VARIABLE(index);
      UNUSED_VARIABLE(pat);
      ERROR_EXIT(128, "Not implemented\n");
    }
    /// Put the pattern bunch
    virtual void putPatternBunch(const int *indexes,unsigned int bunch_size,
                                 Token *pat) {
      UNUSED_VARIABLE(indexes);
      UNUSED_VARIABLE(bunch_size);
      UNUSED_VARIABLE(pat);
      ERROR_EXIT(128, "Not implemented\n");
    }
  };

} // namespace Basics

#endif // UTILDATASETFLOAT_H
