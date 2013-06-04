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
#include "stopwatch.h"

#include <sys/resource.h>
#include <sys/time.h>

namespace april_utils {
  
  double stopwatch::get_cpu_clock() const {
    struct rusage wop;
    getrusage(RUSAGE_SELF, &wop);
    return
      static_cast<double>(wop.ru_utime.tv_sec) + 
      static_cast<double>(wop.ru_utime.tv_usec)*1e-6;
  }

  double stopwatch::get_wall_clock() const {
    struct timeval wop;    
    gettimeofday(&wop, 0);
    return
      static_cast<double>(wop.tv_sec) +
      static_cast<double>(wop.tv_usec)*1e-6;
  }
  
  stopwatch::stopwatch() {
    reset();
    _is_on = false;
  }
  
  void stopwatch::reset() {
    cpu_elapsed  = 0.0;
    wall_elapsed = 0.0;
    cpu_t_ini    = get_cpu_clock();
    wall_t_ini   = get_wall_clock();
  }
  
  void stopwatch::stop() {
    double cpu_increment = get_cpu_clock() - cpu_t_ini;
    if (cpu_increment>0) cpu_elapsed += cpu_increment;
    double wall_increment = get_wall_clock() - wall_t_ini;
    if (wall_increment>0) wall_elapsed += wall_increment;
    _is_on = false;
  }
  
  void stopwatch::go() {
    _is_on = true;
    cpu_t_ini  = get_cpu_clock();
    wall_t_ini = get_wall_clock();
  }
  
  double stopwatch::read_cpu_time() const {
    double increment = (_is_on) ? get_cpu_clock() - cpu_t_ini : 0;
    if (increment<0) increment = 0;
    return cpu_elapsed+increment;
  }

  double stopwatch::read_wall_time() const {
    double increment = (_is_on) ? get_wall_clock() - wall_t_ini : 0;
    if (increment<0) increment = 0;
    return wall_elapsed+increment;
  }

}
