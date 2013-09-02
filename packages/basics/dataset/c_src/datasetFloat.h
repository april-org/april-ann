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
#ifndef UTILDATASETFLOAT_H
#define UTILDATASETFLOAT_H

#include "dataset.h"

typedef DataSet<float> DataSetFloat;
typedef MatrixDataSet<float> MatrixDataSetFloat;
typedef LinearCombConf<float> LinearCombConfFloat;
typedef SparseDataset<float> SparseDatasetFloat;
typedef ShortListDataSet<float> ShortListDataSetFloat;
typedef IndexFilterDataSet<float> IndexFilterDataSetFloat;
typedef PerturbationDataSet<float> PerturbationDataSetFloat;
typedef SaltNoiseDataSet<float> SaltNoiseDataSetFloat;
typedef SaltPepperNoiseDataSet<float> SaltPepperNoiseDataSetFloat;
typedef StepDataSet<float> StepDataSetFloat;
typedef DerivDataSet<float> DerivDataSetFloat;
typedef CacheDataSet<float> CacheDataSetFloat;

#endif // UTILDATASETFLOAT_H
