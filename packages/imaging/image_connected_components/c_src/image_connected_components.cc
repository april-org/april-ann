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

static const float BLACK_THRESHOLD = 0.3;


void ImageConnectedComponents::dfs_component(int x, int y, int current_component){
   
    const int dir = 8;
    int directions[dir][2] = { {-1, -1}, {0, -1}, {1, -1},
                           {-1, 0}, {1, 0},
                           {-1, 1}, {0, 1}, {1, 1}};
   
   int index = y*img->width + x;


   for (int i = 0; i < dir; ++i) {
      int newx = x + directions[i][0];
      int newy = y + directions[i][1];
      int new_index = newy*img->width + newx;
      if (newx < 0 || newx >= img->width || newy < 0 || newy >= img->height || (*img)(newx, newy) >= BLACK_THRESHOLD || pixelComponents[new_index])
         continue;      
      pixelComponents[new_index] = current_component;
      dfs_component(newx, newy, current_component); 
   }  

}


ImageConnectedComponents::ImageConnectedComponents(const ImageFloat *img) {
  
    // 1. Create a Integer Matrix of the same size of the original image
    this->img = img;
    pixelComponents = vector<int>(img->width*img->height, 0); 
    // pixelComponents.fill(0); 
    // 2. Count the number of black pixels
    int black_pixels = img->count_black_pixels(BLACK_THRESHOLD);
    // 3. Create a component of the size black pixels
    components = vector<int>(black_pixels);
    indexComponents = vector<int>();
    
    int current_component = 0;
    // 4. Compute the connected components.
    for (int y = 0; y < img->height; ++y) {
       for (int x = 0; x < img->width; ++x) {
          int index = y*img->width + x;
          float value = (*img)(x,y);
          if ((*img)(x,y) < BLACK_THRESHOLD && !pixelComponents[index]) {
             pixelComponents[index] = ++current_component;
             //DFS over the pixel
             dfs_component(x, y, current_component);
          }
       }
    }

    size = current_component;
}

MatrixInt32 * ImageConnectedComponents::getPixelMatrix(){
  
  int dims[] = {img->height, img->width};
  
  MatrixInt32 *m = new MatrixInt32(2,dims);
  for (int y = 0; y < img->height; ++y) {
      for (int x = 0; x < img->width; ++x) {
          int index = y*img->width + x;
          int value = pixelComponents[index];
          (*m)(y,x) = value;
      }
   }
   return m;
}

