/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
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
#ifndef STOPWATCH_H
#define STOPWATCH_H

#include "referenced.h"

namespace AprilUtils {
  
  class stopwatch : public Referenced {
    double cpu_elapsed; ///< tiempo acumulado de cpu
    double wall_elapsed; ///< tiempo acumulado
    double cpu_t_ini; ///< auxiliar
    double wall_t_ini; ///< auxiliar
    bool _is_on; ///< significado obvio
    /// devuelve tiempo en segundos
    inline double get_cpu_clock() const;
    inline double get_wall_clock() const;
  public:
    stopwatch();
    void reset();
    void stop();
    void go();
    double read_cpu_time() const;
    double read_wall_time() const;
    double read() const { return read_cpu_time(); } // compatible hacia abajo
    bool is_on() const { return _is_on; }
  };

}

#endif // STOPWATCH_H
