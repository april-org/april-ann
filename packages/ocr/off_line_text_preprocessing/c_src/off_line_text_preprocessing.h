/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Salvador España-Boquera, Jorge Gorbe-Moya, Francisco
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
#ifndef OFF_LINE_TEXT_PREPROCESSING_H
#define OFF_LINE_TEXT_PREPROCESSING_H

#include "image.h"
#include "utilImageFloat.h"
#include "pair.h"
#include "vector.h"
#include "geometry.h"

/// Optical Character Recognition stuff.
namespace OCR {
  namespace OffLineTextPreprocessing {

    Basics::MatrixFloat *add_asc_desc (Imaging::ImageFloat     *img,
                                       Basics::MatrixFloat *line_mat,
                                       float v_treshold = 5,
                                       int h_threshold = 22.0f);
    Imaging::ImageFloat *normalize_size(Imaging::ImageFloat *source, float ascender_ratio, float descender_ratio,
                                        AprilUtils::vector<AprilUtils::Point2D> ascenders, AprilUtils::vector<AprilUtils::Point2D> upper_baseline, 
                                        AprilUtils::vector<AprilUtils::Point2D> lower_baseline, AprilUtils::vector<AprilUtils::Point2D> descenders,
                                        int dst_height = -1, bool keep_aspect = false);

    Imaging::ImageFloat *normalize_image(Imaging::ImageFloat *source, int dst_height);

    Imaging::ImageFloat *normalize_size (Imaging::ImageFloat     *source,
                                         Basics::MatrixFloat *line_mat,
                                         float           ascender_ratio,
                                         float           descender_ratio,
                                         int dst_height,
                                         bool keep_aspect);
    AprilUtils::vector<AprilUtils::Point2D>* extract_points_from_image(Imaging::ImageFloat *pimg);

  }
}
#endif
