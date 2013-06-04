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
#ifndef BUFFER_LIST_H
#define BUFFER_LIST_H

#include <stdint.h>
#include "logbase.h"
#include "constString.h"

struct buffer_list_node {
  char *buffer;
  int size;
  buffer_list_node *next;
  buffer_list_node() : buffer(0), size(0), next(0) {};
  buffer_list_node(int sz, buffer_list_node *nxt=0);
  ~buffer_list_node();
};
struct buffer_list {
  enum string_termination { NON_NULL_TERMINATED, NULL_TERMINATED };

  buffer_list_node *first;
  buffer_list_node *last;
  int size;
  //
  buffer_list() : first(0), last(0), size(0) {};
  ~buffer_list();
  void append(buffer_list *other);
  int get_size() const { return size; }

  void add_buffer_left(buffer_list_node *nd);
  void add_buffer_right(buffer_list_node *nd);

  void add_constString_left (const constString &cs);
  void add_constString_right(const constString &cs);

  void add_formatted_string_left (const char *fmt, ...);
  void add_formatted_string_right(const char *fmt, ...);
  /// es una copia de add_formatted_string_right TODO: se puede hacer
  /// que sea un alias en lugar de una copia?
  void printf(const char *fmt, ...);

#ifdef HAVE_UINT16
  void add_binarized_int16_left (const int16_t *v, int v_sz);
  void add_binarized_int16_right(const int16_t *v, int v_sz);

  void add_binarized_uint16_left (const uint16_t *v, int v_sz);
  void add_binarized_uint16_right(const uint16_t *v, int v_sz);
#endif

  void add_binarized_int32_left (const int32_t *v, int v_sz);
  void add_binarized_int32_right(const int32_t *v, int v_sz);

  void add_binarized_uint32_left (const uint32_t *v, int v_sz);
  void add_binarized_uint32_right(const uint32_t *v, int v_sz);

#ifdef HAVE_UINT64
  void add_binarized_int64_left (const int64_t *v, int v_sz);
  void add_binarized_int64_right(const int64_t *v, int v_sz);

  void add_binarized_uint64_left (const uint64_t *v, int v_sz);
  void add_binarized_uint64_right(const uint64_t *v, int v_sz);
#endif

  void add_binarized_float_left (const float *v, int v_sz);
  void add_binarized_float_right(const float *v, int v_sz);

  void add_binarized_log_float_left (const log_float *v, int v_sz);
  void add_binarized_log_float_right(const log_float *v, int v_sz);

  void add_binarized_double_left (const double *v, int v_sz);
  void add_binarized_double_right(const double *v, int v_sz);

  void add_binarized_log_double_left (const log_double *v, int v_sz);
  void add_binarized_log_double_right(const log_double *v, int v_sz);

  char* to_string(string_termination t = NON_NULL_TERMINATED);
};

#endif // BUFFER_LIST_H
