/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco Zamora-Martinez
 * Copyright 2012, Salvador España-Boquera
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
#include <cstdio>
#include "april_print.h"

namespace april_utils {
  void aprilPrint(const float &v) {
    printf("%f", v);
  }
  void aprilPrint(const double &v) {
    printf("%f", v);
  }
  void aprilPrint(const char &v) {
    printf("%c", v);
  }
  void aprilPrint(const int &v) {
    printf("%d", v);
  }
  void aprilPrint(const unsigned int &v) {
    printf("%u", v);
  }
}
