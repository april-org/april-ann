/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
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
#ifndef CBLAS_HEADERS_H
#define CBLAS_HEADERS_H

#ifdef USE_MKL
extern "C" {
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_vml.h>
#include <mkl_service.h>
}
// if you compile with MKL you do not need atlas
#define VECTOR_SSET(n, value, vec, step) for(unsigned int _i_=0;_i_<(n);_i_+=(step))(vec)[_i_]=(value)
#define VECTOR_DSET(n, value, vec, step) for(unsigned int _i_=0;_i_<(n);_i_+=(step))(vec)[_i_]=(value)
#elif USE_XCODE
#include <Accelerate/Accelerate.h>
#include <mm_malloc.h>
#define VECTOR_SSET(n, value, vec, step) for(unsigned int _i_=0;_i_<(n);_i_+=(step))(vec)[_i_]=(value)
#define VECTOR_DSET(n, value, vec, step) for(unsigned int _i_=0;_i_<(n);_i_+=(step))(vec)[_i_]=(value)
#else
//#error "JARL"
extern "C" {
#include <atlas/cblas.h>
}
#include <mm_malloc.h>
#define VECTOR_SSET(n, value, vec, step) catlas_sset((n), (value), (vec), (step))
#define VECTOR_DSET(n, value, vec, step) catlas_dset((n), (value), (vec), (step))
#endif

#define NEGATE_CBLAS_TRANSPOSE(trans) ((trans) == CblasNoTrans)?CblasTrans:CblasNoTrans

#endif // CBLAS_HEADERS_H
