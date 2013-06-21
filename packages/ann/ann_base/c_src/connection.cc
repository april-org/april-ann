/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
#include "swap.h"
#include "connection.h"
#include "check_floats.h"
#include "wrapper.h"

namespace ANN {
  const double Connections::weightnearzero = 1e-7;
  
  Connections::Connections(unsigned int num_inputs, unsigned int num_outputs) :
    Referenced(),
    weights(0), prev_weights(0),
    num_references(0), update_weights_calls(0) {
    int dims[2] = { static_cast<int>(num_outputs),
		    static_cast<int>(num_inputs) };
    weights      = new MatrixFloat(2, dims, 0.0f, CblasColMajor);
    prev_weights = new MatrixFloat(2, dims, 0.0f, CblasColMajor);
    if (weights == 0 || prev_weights == 0)
      ERROR_EXIT(130, "Impossible to allocate memory\n");
    IncRef(weights);
    IncRef(prev_weights);
    weights->zeros();
    prev_weights->zeros();
  }

  Connections::~Connections() {
    DecRef(weights);
    DecRef(prev_weights);
  }

  bool Connections::checkInputOutputSizes(unsigned int input_size,
					  unsigned int output_size) const {
    // TODO: comprobar error input==0 y output==0
    if (static_cast<unsigned int>(weights->getDimSize(0)) != output_size) {
      ERROR_PRINT("Incorrect output size!!!\n");
      return false;
    }
    if (static_cast<unsigned int>(weights->getDimSize(1)) != input_size) {
      ERROR_PRINT("Incorrect input size!!!\n");
      return false;
    }
    return true;
  }

  void Connections::countReference() {
    ++num_references;
  }
    
  unsigned int Connections::getNumReferences() const {
    return num_references;
  }
    
  void Connections::beginUpdate() {
    ++update_weights_calls;
  }
    
  bool Connections::endUpdate() {
    // if it is the last call
    if (update_weights_calls == num_references) {
      // Swap(w, prev_w)
      april_utils::swap(weights, prev_weights);
      update_weights_calls = 0;
      return true;
    }
    return false;
  }
    
  bool Connections::isFirstUpdateCall() {
    return update_weights_calls == 1;
  }

  void Connections::
  computeMomentumOnPrevVector(float momentum, bool use_cuda) {
#ifdef USE_CUDA
    // FIXME: adds a setUseCuda method to set CUDA flag of the matrices
    weights->setUseCuda(use_cuda);
    prev_weights->setUseCuda(use_cuda);
#endif
    // momentum learning rule
    // prev_w[i,j] = momentum * (w[i,j] - prev_w[i,j])
    //
    // but this method computes: first the complementary with saxpy:
    // prev_w[i,j] = prev_w[i,j] - 1.0f * w[i,j]
    prev_weights->axpy(-1.0f, weights);
    // second apply momentum with sscal:
    // prev_w[i,j] = -momentum * prev_w[i,j] = -momentum*(prev_w[i,j] - w[i,j])
    prev_weights->scal(-momentum);
  }
  
  void Connections::
  computeWeightDecayOnPrevVector(float c_weight_decay, bool use_cuda) {
#ifdef USE_CUDA
    // FIXME: adds a setUseCuda method to set CUDA flag of the matrices
    weights->setUseCuda(use_cuda);
    prev_weights->setUseCuda(use_cuda);
#endif
    // applies weight decay
    // prev_w[i,j] = c_weight_decay * w[i,j] + prev_w[i,j]
    //
    prev_weights->axpy(c_weight_decay, weights);
  }

  unsigned int Connections::size() const {
    return weights->size();
  }
    
  void Connections::copyToPrevVector(bool use_cuda) {
#ifdef USE_CUDA
    // FIXME: adds a setUseCuda method to set CUDA flag of the matrices
    weights->setUseCuda(use_cuda);
    prev_weights->setUseCuda(use_cuda);
#endif
    prev_weights->copy(weights);
  }
  
  void Connections::pruneSubnormalAndCheckNormal() {
    float *w = weights->getRawDataAccess()->getPPALForReadAndWrite();
    if (!april_utils::check_floats(w, weights->size())) {
      assert("No finite numbers at weights matrix!!!" && false);
      ERROR_EXIT(128, "No finite numbers at weights matrix!!!\n");
    }
  }

  MatrixFloat *Connections::getPtr() {
    return weights;
  }

  MatrixFloat *Connections::getPrevPtr() {
    return prev_weights;
  }

  // Crea de forma aleatoria el conjunto de pesos con valores en el
  // rango [low, high]
  void Connections::randomizeWeights(MTRand *rnd, float low, float high) {
    double dinf = low;
    double dsup = high;

    // assert to avoid nearzero weights
    assert(fabs(dinf) > weightnearzero);
    assert(fabs(dsup) > weightnearzero);
    double range  = dsup - dinf;
    MatrixFloat::iterator w_it(weights->begin());
    MatrixFloat::iterator prev_w_it(prev_weights->begin());
    while(w_it != weights->end()) {
      rnd_weight(rnd, *w_it, dinf, range, weightnearzero);
      *prev_w_it = *w_it;
      ++w_it;
      ++prev_w_it;
    }
  }
    
  void Connections::randomizeWeightsAtColumn(unsigned int col,
					     MTRand *rnd,
					     float low, float high) {
    double dinf = low;
    double dsup = high;

    // assert to avoid nearzero weights
    assert(fabs(dinf) > weightnearzero);
    assert(fabs(dsup) > weightnearzero);
    double range  = dsup - dinf;
    MatrixFloat::iterator w_it(weights->iteratorAt(col,0));
    MatrixFloat::iterator prev_w_it(prev_weights->iteratorAt(col,0));
    MatrixFloat::iterator end(weights->iteratorAt(col+1,0));
    while(w_it != end) {
      rnd_weight(rnd, *w_it, dinf, range, weightnearzero);
      *prev_w_it = *w_it;
      ++w_it;
      ++prev_w_it;
    }
  }
  
  unsigned int Connections::loadWeights(MatrixFloat *data,
					MatrixFloat *old_data,
					unsigned int first_weight_pos,
					unsigned int column_size) {
    const unsigned int num_outputs = static_cast<unsigned int>(weights->getDimSize(0));
    const unsigned int num_inputs  = static_cast<unsigned int>(weights->getDimSize(1));
    const unsigned int total_size  = static_cast<unsigned int>(weights->size());
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size()))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size(), min_size);
    if (!old_data) old_data = data;
    if (!data->isSimple() || !old_data->isSimple())
      ERROR_EXIT(128, "Matrices need to be simple (contiguous "
		 "and in row-major)\n");
    if (data->getNumDim() != old_data->getNumDim())
      ERROR_EXIT(128, "data and old_data has different number of dimensions\n");
    
    unsigned int current_w_pos = first_weight_pos;
    MatrixFloat::iterator w_it(weights->begin());
    MatrixFloat::iterator prev_w_it(prev_weights->begin());
    for (unsigned int j=0; j<num_outputs; ++j) {
      for (unsigned int i=0; i<num_inputs; ++i) {
	*w_it      = (*data)[current_w_pos+i];
	*prev_w_it = (*old_data)[current_w_pos+i];
	++w_it;
	++prev_w_it;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }

  unsigned int Connections::copyWeightsTo(MatrixFloat *data,
					  MatrixFloat *old_data,
					  unsigned int first_weight_pos,
					  unsigned int column_size) {
    const unsigned int num_outputs = static_cast<unsigned int>(weights->getDimSize(0));
    const unsigned int num_inputs  = static_cast<unsigned int>(weights->getDimSize(1));
    const unsigned int total_size  = static_cast<unsigned int>(weights->size());
    unsigned int min_size =
      (total_size +
       max(0, (static_cast<int>(column_size-num_inputs)-1))*num_outputs +
       first_weight_pos);
    if (min_size > static_cast<unsigned int>(data->size()))
      ERROR_EXIT2(24, "Incorrect matrix size, was %d, expected >= %d\n",
		  data->size(), min_size);
    if (!data->isSimple() || !old_data->isSimple())
      ERROR_EXIT(128, "Matrices need to be simple (contiguous "
		 "and in row-major)\n");
    if (data->getNumDim() != old_data->getNumDim())
      ERROR_EXIT(128, "data and old_data has different number of dimensions\n");
    
    unsigned int current_w_pos = first_weight_pos;
    MatrixFloat::const_iterator w_it(weights->begin());
    MatrixFloat::const_iterator prev_w_it(prev_weights->begin());
    for (unsigned int j=0; j<num_outputs; ++j) {
      for (unsigned int i=0; i<num_inputs; ++i) {
	(*data)[current_w_pos+i]     = *w_it;
	(*old_data)[current_w_pos+i] = *prev_w_it;
	++w_it;
	++prev_w_it;
      }
      current_w_pos += column_size;
    }
    return current_w_pos;
  }
    
  // para hacer copias
  Connections *Connections::clone() {
    const int num_outputs = weights->getDimSize(0);
    const int num_inputs  = weights->getDimSize(1);
    Connections *conn = new Connections(num_inputs, num_outputs);
    conn->weights->copy(weights);
    conn->prev_weights->copy(prev_weights);
    return conn;
  }

  void Connections::scale(float alpha) {
    weights->scal(alpha);
    prev_weights->scal(alpha);
  }
  
  void Connections::printDebug() {
    const int num_outputs = weights->getDimSize(0);
    const int num_inputs  = weights->getDimSize(1);
    printf ("Connections %p, input=%d, output=%d, num_refs=%d, calls=%d\n",
	    this, num_inputs, num_outputs, num_references,
	    update_weights_calls);
    for (MatrixFloat::iterator w_it(weights->begin()); w_it!=weights->end();
	 ++w_it) 
      printf("%f ", *w_it);
    printf("\n");
    for (MatrixFloat::iterator prev_w_it(prev_weights->begin());
	 prev_w_it!=prev_weights->end();
	 ++prev_w_it) 
      printf("%f ", *prev_w_it);
    printf("\n");
  }
  
}
