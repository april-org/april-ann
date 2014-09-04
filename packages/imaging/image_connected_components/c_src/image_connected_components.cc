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

#include "image_connected_components.h"

using namespace AprilUtils;
using namespace Basics;

namespace Imaging {

  namespace rgb_colors {
    typedef enum { RED, GREEN, BLUE, YELLOW, ORANGE, PURPLE, PINK, BROWN, BLACK } COLOR;

    float colors[BLACK+1][3] = {
      { 1, 0, 0},
      { 0, 1, 0},
      { 0, 0, 1},
      { 0, 1, 1},
      { 1, 0.5, 1},
      {1, 1, 0},
      {1, 0, 1},
      {0.4, 0.2, 0.5},
      {0, 0, 0},

    };

    float * getIndexColor(int index) {
      return colors[index%BLACK];
    }
  }

  inline int to_index(const ImageFloat *img, int x, int y) {
    return y*img->width() + x;
  }

  inline void to_coords(const ImageFloat *img, int index, int &x, int &y) {
    y = index/img->width();
    x = index%img->width();
  }

  void ImageConnectedComponents::dfs_component(int x, int y, int current_component, int &current_pixel){
   
    const int dir = 8;
    int directions[dir][2] = { {-1, -1}, {0, -1}, {1, -1},
                               {-1, 0}, {1, 0},
                               {-1, 1}, {0, 1}, {1, 1}};
   
    for (int i = 0; i < dir; ++i) {
      int newx = x + directions[i][0];
      int newy = y + directions[i][1];
      int new_index = to_index(img, newx, newy);
      if (newx < 0 || newx >= img->width() || newy < 0 || newy >= img->height() || pixelComponents[new_index])
        continue;      
      if ((*img)(newx, newy) < threshold) {// If it is black
        components[current_pixel++] = new_index; 
      }
      if ((*img)(x, y) < threshold || (*img)(newx, newy) < threshold) {
        pixelComponents[new_index]  = current_component;
        dfs_component(newx, newy, current_component, current_pixel); 
      }
    }  

  }

  ImageConnectedComponents::ImageConnectedComponents(const ImageFloat *img, float threshold):threshold(threshold) {
    // 1. Create a Integer Matrix of the same size of the original image
    this->img = img;
    pixelComponents = vector<int>(img->width()*img->height(), 0);

    // 2. Count the number of black pixels
    int black_pixels = img->count_black_pixels(threshold);

    // 3. Create a component of the size black pixels
    components = vector<int>(black_pixels);
    indexComponents = vector<int>();


    int current_component = 0;
    int current_pixel     = 0;
    // 4. Compute the connected components.
    for (int y = 0; y < img->height(); ++y) {
      for (int x = 0; x < img->width(); ++x) {
        int index = to_index(img, x, y);
        if ((*img)(x,y) < threshold && !pixelComponents[index]) {
          pixelComponents[index] = ++current_component;
          components[current_pixel++] = index;
          //DFS over the pixel
          dfs_component(x, y, current_component, current_pixel);
          indexComponents.push_back(current_pixel);
        }
      }
    }

    size = current_component;

  }

  MatrixInt32 * ImageConnectedComponents::getPixelMatrix(){

    int dims[] = {img->height(), img->width()};

    MatrixInt32 *m = new MatrixInt32(2,dims);
    for (int y = 0; y < img->height(); ++y) {
      for (int x = 0; x < img->width(); ++x) {
        int index = to_index(img, x, y);
        int value = pixelComponents[index];
        (*m)(y,x) = value;
      }
    }
    return m;
  }

  ImageFloatRGB *ImageConnectedComponents::getColoredImage() {

    using rgb_colors::getIndexColor;

    Matrix<FloatRGB> *m = new Matrix<FloatRGB>(2, img->height(), img->width());
    AprilMath::MatrixExt::Operations::matFill(m, FloatRGB(1, 1, 1));
    ImageFloatRGB *result = new ImageFloatRGB(m);

    int first = 0;
    for (int cc = 0; cc < size; ++cc) {
      //    printf("Component %d %d\n", cc, indexComponents[cc]); 
      for (int i = first; i < indexComponents[cc]; ++i) {
        float *color = getIndexColor(cc);
        int index = components[i];
        int x, y;
        to_coords(img, index, x, y);
        (*m)(y,x) = FloatRGB(color[0], color[1], color[2]);
      }
      first = indexComponents[cc];
    }

    return result;
  }

  bool ImageConnectedComponents::connected(int x1, int y1, int x2, int y2) {

    april_assert(x1 <= img->width() && x1 > 0 && y1 <= img->height() && y1 > 0 && "X1, Y1 point out of bounds");
    april_assert(x2 <= img->width() && x2 > 0 && y2 <= img->height() && y2 > 0 && "X2, Y2 point out of bounds");

    int index1 = to_index(img, x1, y1);
    int index2 = to_index(img, x2, y2);

    if (!pixelComponents[index1] && !pixelComponents[index2]) {
      fprintf(stderr, "Warning! One of the point is not in a component\n");
      return false;
    }

    return pixelComponents[index1] == pixelComponents[index2]; 
  }

  int ImageConnectedComponents::getComponent(int x, int y) {
    int index = to_index(img, x, y);
    return pixelComponents[index] -1;
  }

  bounding_box ImageConnectedComponents::getComponentBoundingBox(int component) {
    assert(component >= 0 && component < size && "The component is not corrected"); 
  
    int from = component ? indexComponents[component-1] : 0;
    int to = indexComponents[component];

    int x1 = img->width();
    int y1 = img->height();
    int x2 = 0;
    int y2 = 0;
    for (int i = from; i < to; ++i) {
      int x, y;
      int pointIndex = components[i];
      to_coords(img, pointIndex, x, y);
      x1 = min(x1, x);
      y1 = min(y1, y);

      x2 = max(x2, x);
      y2 = max(y2, y);
    }

    return bounding_box(x1,y1, x2, y2);
  }

  vector<bounding_box> * ImageConnectedComponents::getBoundingBoxes() {
  
    vector<bounding_box> *bbs = new vector<bounding_box>();
    for (int i = 0; i < size; ++i) {
      bbs->push_back(getComponentBoundingBox(i));
    }

    return bbs;
  }

} // namespace Imaging
