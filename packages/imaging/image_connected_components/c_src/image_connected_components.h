/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
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
#ifndef IMAGE_CONNECTED_COMPONENTS_H
#define IMAGE_CONNECTED_COMPONENTS_H

#include "matrixInt32.h"
#include "utilImageFloat.h"
#include "vector.h"

using namespace april_utils;

class ImageConnectedComponents: public Referenced{

  // Matrix of the size of the image that is used to 
  vector <int> pixelComponents;
  vector <int> components;
  vector <int> indexComponents;
  const ImageFloat *img; 
  public:
      int size;
      ImageConnectedComponents(const ImageFloat *img);
      ~ImageConnectedComponents(){};
      MatrixInt32 *getPixelMatrix();
  private:
      void dfs_component(int x, int y, int current_component);
};
#endif
