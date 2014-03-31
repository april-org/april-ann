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
#include "cusparse_error.h"

#ifdef USE_CUDA

void checkCusparseError(cublasStatus_t status) {
  if (status == CUSPARSE_STATUS_SUCCESS)
    return;
  else if (status == CUSPARSE_STATUS_NOT_INITIALIZED)
    ERROR_EXIT(152, "Cusparse is not initialized!\n");
  else if (status == CUSPARSE_STATUS_ALLOC_FAILED)
    ERROR_EXIT(153, "Cusparse resource allocation failed!\n");
  else if (status == CUSPARSE_STATUS_INVALID_VALUE)
    ERROR_EXIT(154, "Cusparse detected an unsupported parameter.\n");
  else if (status == CUSPARSE_STATUS_ARCH_MISMATCH)
    ERROR_EXIT(155, "Cusparse and architecture not compatible.\n");
  else if (status == CUSPARSE_STATUS_MAPPING_ERROR)
    ERROR_EXIT(156, "Cusparse accessed to a wrong GPU memory space!\n");
  else if (status == CUSPARSE_STATUS_EXECUTION_FAILED)
    ERROR_EXIT(157, "Cusparse failed to execute the operation\n");
  else if (status == CUSPARSE_STATUS_INTERNAL_ERROR)
    ERROR_EXIT(158, "Cusparse failed to execute an internal operation!\n");
}

#endif
