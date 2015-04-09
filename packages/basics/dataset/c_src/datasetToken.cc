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

#include "dataset.h"
#include "datasetFloat.h"
#include "datasetToken.h"
#include "function_interface.h"
#include "matrixFloat.h"
#include "matrix_ext.h"
#include "smart_ptr.h"
#include "token_base.h"
#include "token_matrix.h"
#include "token_sparse_matrix.h"
#include "token_vector.h"
#include "table_of_token_codes.h"
#include "unused_variable.h"
#include "vector.h"

using AprilMath::MatrixExt::BLAS::matCopy;
using AprilUtils::SharedPtr;
using AprilUtils::vector;
using AprilMath::FloatGPUMirroredMemoryBlock;
using AprilMath::Int32GPUMirroredMemoryBlock;

namespace Basics {
  
  namespace DataSetTokenUtils {
    
    SharedPtr<Token> buildMatrixFloatBunch(DataSetToken *ds,
                                           const int *indexes,
                                           unsigned int bunch_size,
                                           SharedPtr<Token> aux_token) {
      if (aux_token.empty()) aux_token = ds->getPattern(indexes[0]);
      // FIXME: Check this shape to be consistent when getPattern() returns a multi-dimensional matrix.
      int dims[2]   = { static_cast<int>(bunch_size), ds->patternSize() };
      SharedPtr<MatrixFloat> output_mat( new MatrixFloat(2, dims) );
      TokenMatrixFloat *output_mat_token = new TokenMatrixFloat(output_mat.get());
      SharedPtr<Token> result( output_mat_token );
      // the dims vector will be used to rewrap the matrices to have the same
      // structure
      dims[0] = ds->patternSize();
      SharedPtr<MatrixFloat> submat;
      for (int i=0; i<static_cast<int>(bunch_size); ++i) {
        if (i>0) aux_token = ds->getPattern(indexes[i]);
        TokenMatrixFloat *mat_token = aux_token->convertTo<TokenMatrixFloat*>();
        SharedPtr<MatrixFloat> mat( mat_token->getMatrix() );
        SharedPtr<MatrixFloat> mat_rewrapped( mat->rewrap(dims, 1) );
        submat = output_mat->select(0, i, submat.get());
        matCopy(submat.get(), mat_rewrapped.get());
      
      }
      return result;
    }
    
    SharedPtr<Token> buildSparseMatrixFloatBunch(DataSetToken *ds,
                                                 const int *indexes,
                                                 unsigned int bunch_size,
                                                 SharedPtr<Token> aux_token) {
      if (aux_token.empty()) aux_token = ds->getPattern(indexes[0]);
      TokenSparseMatrixFloat *sparse_token =
        aux_token->convertTo<TokenSparseMatrixFloat*>();
      SparseMatrixFloat *sparse_matrix = sparse_token->getMatrix();
      int dims[2] = { static_cast<int>(bunch_size), ds->patternSize() };
      unsigned int nnz = sparse_matrix->nonZeroSize();
      vector< SharedPtr<SparseMatrixFloat> > matrices(bunch_size);
      matrices[0] = sparse_matrix;
      for (unsigned int i=1; i<bunch_size; ++i) {
        aux_token = ds->getPattern(indexes[i]);
        sparse_token = aux_token->convertTo<TokenSparseMatrixFloat*>();
        sparse_matrix = sparse_token->getMatrix();
        matrices[i] = sparse_matrix;
        nnz += sparse_matrix->nonZeroSize();
      }
      // build a sparse matrix with nnz values in CSR format
      SharedPtr< FloatGPUMirroredMemoryBlock >
        values(new FloatGPUMirroredMemoryBlock(nnz));
      SharedPtr< Int32GPUMirroredMemoryBlock >
        indices(new Int32GPUMirroredMemoryBlock(nnz));
      SharedPtr< Int32GPUMirroredMemoryBlock >
        first_index(new Int32GPUMirroredMemoryBlock(bunch_size + 1));
      (*first_index)[0] = 0;
      size_t pos=0;
      for (unsigned int i=0; i<bunch_size; ++i) {
        sparse_matrix = matrices[i].get();
        unsigned int sz = sparse_matrix->nonZeroSize();
        if (sparse_matrix->getSparseFormat() != CSR_FORMAT) {
          ERROR_EXIT(128, "Only implemented for CSR sparse matrices");
        }
        values->copyFromBlock(pos, sparse_matrix->getRawValuesAccess(), 0, sz);
        indices->copyFromBlock(pos, sparse_matrix->getRawIndicesAccess(), 0, sz);
        pos += sz;
        (*first_index)[i+1] = pos;
      }
      SparseMatrixFloat *result = new SparseMatrixFloat(dims[0], dims[1],
                                                        values.get(),
                                                        indices.get(),
                                                        first_index.get());
      return new TokenSparseMatrixFloat(result);
    }
  
    // SharedPtr<Token> buildTokenVectorBunch(DataSetToken *ds,
    //                                        const int *indexes,
    //                                        unsigned int bunch_size)
    // {
    //   TokenBunchVector *result_token_vector = new TokenBunchVector(bunch_size);
    //   SharedPtr<Token> result( result_token_vector );
    //   for (unsigned int i = 0; i<bunch_size; ++i) {
    //     Token *tk = ds->getPattern(indexes[i]);
    //     TokenCode token_code = tk->getTokenCode();
    //     SharedPtr<Token> current;
    //     switch(token_code) {
    //     case table_of_token_codes::token_matrix:
    //       current = buildMatrixFloatBunch(ds, indexes, bunch_size);
    //       break;
    //     case table_of_token_codes::token_sparse_matrix:
    //       current = buildSparseMatrixFloatBunch(ds, indexes, bunch_size);
    //       break;
    //     default:
    //       current = tk;
    //       break;
    //     } 
    //     (*result_token_vector)[i] = current.weakRelease();
    //   }
    //   return result;
    // }

  } // namespace DataSetTokenUtils
  
} // namespace Basics

using namespace Basics::DataSetTokenUtils;

namespace Basics {

  /// Get the pattern index to the vector pat
  Token *DataSetToken::getPatternBunch(const int *indexes,
                                       unsigned int bunch_size) {
    SharedPtr<Token> result;
    SharedPtr<Token> aux_token( getPattern(indexes[0]) );
    TokenCode token_code = aux_token->getTokenCode();
    switch(token_code) {
    case table_of_token_codes::token_matrix:
      result = buildMatrixFloatBunch(this, indexes, bunch_size, aux_token);
      break;
    case table_of_token_codes::token_sparse_matrix:
      result = buildSparseMatrixFloatBunch(this, indexes, bunch_size, aux_token);
      break;
    default:
      ERROR_EXIT(128, "Not implemented\n");
      // result = buildTokenVectorBunch(this, indexes, bunch_size);
      break;
    }
    return result.weakRelease();
  }
  
  Token *SparseMatrixDataSetToken::getPatternBunch(const int *indexes,
                                                   unsigned int bunch_size) {
    unsigned int nnz = 0;
    const FloatGPUMirroredMemoryBlock *data_values = data->getRawValuesAccess();
    const Int32GPUMirroredMemoryBlock *data_indices = data->getRawIndicesAccess();
    const Int32GPUMirroredMemoryBlock *data_first_index = data->getRawFirstIndexAccess();
    for (unsigned int i=0; i<bunch_size; ++i) {
      const int p = indexes[i];
      if (p < 0 || p >= numPatterns()) {
        ERROR_EXIT2(128, "Out-of-bounds index, expected in "
                    "range [0,%d], found %d\n", numPatterns(), p);
        
      }
      nnz += (*data_first_index)[p+1] - (*data_first_index)[p];
    }
    int dims[2] = { static_cast<int>(bunch_size), patternSize() };
    // build a sparse matrix with nnz values in CSR format
    SharedPtr< FloatGPUMirroredMemoryBlock >
      values(new FloatGPUMirroredMemoryBlock(nnz));
    SharedPtr< Int32GPUMirroredMemoryBlock >
      indices(new Int32GPUMirroredMemoryBlock(nnz));
    SharedPtr< Int32GPUMirroredMemoryBlock >
      first_index(new Int32GPUMirroredMemoryBlock(bunch_size + 1));
    (*first_index)[0] = 0;
    size_t pos=0;
    for (unsigned int i=0; i<bunch_size; ++i) {
      int p = indexes[i];
      unsigned int from = (*data_first_index)[p];
      unsigned int sz = (*data_first_index)[p+1] - (*data_first_index)[p];
      values->copyFromBlock(pos, data_values, from, sz);
      indices->copyFromBlock(pos, data_indices, from, sz);
      pos += sz;
      (*first_index)[i+1] = pos;
    }
    SparseMatrixFloat *result = new SparseMatrixFloat(dims[0], dims[1],
                                                      values.get(),
                                                      indices.get(),
                                                      first_index.get());
    return new TokenSparseMatrixFloat(result);
  }
  
} // namespace Basics
