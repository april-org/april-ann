 /*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Francisco Zamora-Martinez
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
#ifndef LAPACK_HEADERS_H
#define LAPACK_HEADERS_H

#ifdef USE_MKL
/////////////////////////////////// MKL ///////////////////////////////////////
extern "C" {
#include <mkl_lapack.h>
}
/*****************************************************************************/
#elif USE_XCODE
////////////////////////////////// XCODE //////////////////////////////////////
#include <Accelerate/Accelerate.h>
/*****************************************************************************/
#else
////////////////////////////////// ATLAS //////////////////////////////////////
extern "C" {
#include <atlas/lapack.h>
}
/*****************************************************************************/
#endif

void checkLapackInfo(int info);

#endif // LAPACK_HEADERS_H
 
