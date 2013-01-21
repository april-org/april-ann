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
#ifndef ANNCONFIGURATION_H
#define ANNCONFIGURATION_H

// information required by different parts of the ann grouped into a public class
// a constant reference is keeped by them, the object belongs to the ann
struct ANNConfiguration {
  unsigned int max_bunch_size; // maximum bunch size ANTES num_bunch
  unsigned int cur_bunch_size; // current bunch size, ANTES cur_bunch
  bool         use_cuda_flag; // use_cuda
  bool         output_error_in_logbase;
  ANNConfiguration(int max_bunch_size,
		   int cur_bunch_size,
		   bool use_cuda_flag=false,
		   bool output_error_in_logbase=false) :
    max_bunch_size(max_bunch_size),
    cur_bunch_size(cur_bunch_size),
    use_cuda_flag(use_cuda_flag),
    output_error_in_logbase(output_error_in_logbase) {
  }
};

#endif
