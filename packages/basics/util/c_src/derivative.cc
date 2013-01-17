/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Jorge Gorbe Moya, Francisco Zamora-Martinez
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
#include "derivative.h"
#include "clamp.h"

namespace april_utils {

  // Explicacion de las formulas en:
  // http://legacy.ncsu.edu/classes-a/mat310_info/DiffInt/NumDeriv.html
  void derivative1(float *v, float *v_dest, int n)
  {
    int izq1, izq2, der1, der2;

    for (int i=0; i<n; ++i)
    {
      izq1 = clamp(i-1, 0, n-1);
      izq2 = clamp(i-2, 0, n-1);
      der1 = clamp(i+1, 0, n-1);
      der2 = clamp(i+2, 0, n-1);

      v_dest[i] = (-v[der2] + 8*v[der1] - 8*v[izq1] + v[izq2])/12.0;
    }
  }

  void derivative2(float *v, float *v_dest, int n)
  {
    int izq1, izq2, der1, der2;

    for (int i=0; i<n; ++i)
    {
      izq1 = clamp(i-1, 0, n-1);
      izq2 = clamp(i-2, 0, n-1);
      der1 = clamp(i+1, 0, n-1);
      der2 = clamp(i+2, 0, n-1);

      v_dest[i] = (-v[der2] + 16*v[der1] - 30*v[i] + 16*v[izq1] - v[izq2])/12.0;
    }

  }

}

