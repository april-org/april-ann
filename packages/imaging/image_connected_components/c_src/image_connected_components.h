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

namespace Imaging {

  struct bounding_box {
    int x1, y1;
    int x2, y2;

    bounding_box(){};
    bounding_box(int x1, int y1, int x2, int y2):
      x1(x1),y1(y1),x2(x2),y2(y2){}
  };

  class ImageConnectedComponents: public Referenced{

    // Matrix of the size of the image that is used to 
    AprilUtils::vector <int> pixelComponents;

    // Vector that stores the black pixels sorted by components 
    AprilUtils::vector <int> components;

    // index that delimites the CCs in components
    AprilUtils::vector <int> indexComponents;

    //black threshold
    float threshold;
    const ImageFloat *img;
  public:
    int size;
    ImageConnectedComponents(const ImageFloat *img, float threshold = 0.7);
    ~ImageConnectedComponents(){};

  private:
    void dfs_component(int x, int y, int current_component, int &current_pixel);

  public:
    Basics::MatrixInt32 *getPixelMatrix();
    bool connected(int x1, int y1, int x2, int y2);

    int getComponent(int x, int y);
    ImageFloatRGB  *getColoredImage();
    bounding_box getComponentBoundingBox(int component);
    AprilUtils::vector<bounding_box> *getBoundingBoxes();    

  };

} // namespace Imaging

#endif
