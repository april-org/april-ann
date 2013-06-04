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
#include "buffer_list.h"
#include <cstdarg>
#include <cassert>
#include <cstdio>
#include <cstring>
#include "binarizer.h"

buffer_list_node::buffer_list_node(int sz, buffer_list_node *nxt) {
  size   = sz;
  buffer = new char[size];
  next   = nxt;
}

buffer_list_node::~buffer_list_node() {
  if (buffer) delete[] buffer;
}

buffer_list::~buffer_list() {
  while (first) {
    buffer_list_node *aux = first;
    first = first->next;
    delete aux;
  }
}

void buffer_list::append(buffer_list *other) {
  size += other->size;
  if (last) {
    last->next = other->first;
  } else { // lista izquierda está vacía
    first = other->first;
  }
  last = other->last;

  // Los nodos de other pasan a ser de this.
  // Por tanto, other esta vacio
  other->first = 0;
  other->last = 0;
  other->size = 0;
}

void buffer_list::add_buffer_left(buffer_list_node *nd) {
  size += nd->size;
  nd->next = first;
  first = nd;
  if (!last) last=first;
}

void buffer_list::add_buffer_right(buffer_list_node *nd) {
  size += nd->size;
  nd->next = 0;
  if (last) {
    last->next = nd;
  } else {
    first = nd;
  }
  last = nd;
}

#define FORMAT_BUFFER_SIZE 200

void buffer_list::add_formatted_string_left(const char *fmt, ...) {
  int n;
  char buf[FORMAT_BUFFER_SIZE];
  buffer_list_node *node;
  va_list ap;
  
  va_start(ap, fmt);
  n = vsnprintf (buf, FORMAT_BUFFER_SIZE, fmt, ap);
  va_end(ap);

  assert(n>=0);

  // n doesn't include the terminating '\0'
  node = new buffer_list_node(n+1,0);

  if (n < FORMAT_BUFFER_SIZE) {
    strcpy(node->buffer, buf);
  } else {
    va_start(ap, fmt);
    n = vsnprintf (node->buffer, FORMAT_BUFFER_SIZE, fmt, ap);
    va_end(ap);
  }

  add_buffer_left(node);
}

void buffer_list::add_formatted_string_right(const char *fmt, ...) {
  int n;
  char buf[FORMAT_BUFFER_SIZE];
  buffer_list_node *node;
  va_list ap;
  
  va_start(ap, fmt);
  n = vsnprintf (buf, FORMAT_BUFFER_SIZE, fmt, ap);
  va_end(ap);

  assert(n>=0);

  // n doesn't include the terminating '\0'
  // but we don't want a null-terminated string anyway
  node = new buffer_list_node(n,0);

  if (n < FORMAT_BUFFER_SIZE) {
    memcpy(node->buffer, buf, n);
  } else {
    va_start(ap, fmt);
    n = vsnprintf (node->buffer, n, fmt, ap);
    va_end(ap);
  }

  add_buffer_right(node);
}

void buffer_list::printf(const char *fmt, ...) {
  int n;
  char buf[FORMAT_BUFFER_SIZE];
  buffer_list_node *node;
  va_list ap;
  
  va_start(ap, fmt);
  n = vsnprintf (buf, FORMAT_BUFFER_SIZE, fmt, ap);
  va_end(ap);

  assert(n>=0);

  // n doesn't include the terminating '\0'
  // but we don't want a null-terminated string anyway
  node = new buffer_list_node(n,0);

  if (n < FORMAT_BUFFER_SIZE) {
    memcpy(node->buffer, buf, n);
  } else {
    va_start(ap, fmt);
    n = vsnprintf (node->buffer, n, fmt, ap);
    va_end(ap);
  }

  add_buffer_right(node);
}

void buffer_list::add_constString_left (const constString &cs) {
  buffer_list_node *node = new buffer_list_node(cs.len());
  memcpy(node->buffer, (const char *)cs, cs.len());
  add_buffer_left(node);
}

void buffer_list::add_constString_right (const constString &cs) {
  buffer_list_node *node = new buffer_list_node(cs.len());
  memcpy(node->buffer, (const char *)cs, cs.len());
  add_buffer_right(node);
}


char *buffer_list::to_string(string_termination t) {
  int s = size;

  if (t == NULL_TERMINATED) {
    s++;
  } 

  char *res = new char[s];
  char *pos = res;
  buffer_list_node *cur;

  for (cur = first; cur != NULL; cur = cur->next) {
    memcpy(pos, cur->buffer, cur->size);
    pos += cur->size;
  }

  if (t==NULL_TERMINATED) {
    *pos = '\0';
  }

  return res;
}

#ifdef HAVE_UINT16
void buffer_list::add_binarized_int16_left (const int16_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*3);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=3)
    binarizer::code_int16(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_int16_right (const int16_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*3);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=3)
    binarizer::code_int16(v[i],r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_uint16_left (const uint16_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*3);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=3)
    binarizer::code_uint16(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_uint16_right (const uint16_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*3);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=3)
    binarizer::code_uint16(v[i],r);
  add_buffer_right(node);
}
#endif

void buffer_list::add_binarized_int32_left (const int32_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_int32(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_int32_right (const int32_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_int32(v[i],r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_uint32_left (const uint32_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_uint32(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_uint32_right (const uint32_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_uint32(v[i],r);
  add_buffer_right(node);
}

#ifdef HAVE_UINT64
void buffer_list::add_binarized_int64_left (const int64_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_int64(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_int64_right (const int64_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_int64(v[i],r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_uint64_left (const uint64_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_uint64(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_uint64_right (const uint64_t *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_uint64(v[i],r);
  add_buffer_right(node);
}
#endif

void buffer_list::add_binarized_float_left (const float *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_float(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_float_right (const float *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_float(v[i],r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_log_float_left (const log_float *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_float(v[i].log(),r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_log_float_right (const log_float *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*5);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=5)
    binarizer::code_float(v[i].log(),r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_double_left (const double *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_double(v[i],r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_double_right (const double *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_double(v[i],r);
  add_buffer_right(node);
}

void buffer_list::add_binarized_log_double_left (const log_double *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_double(v[i].log(),r);
  add_buffer_left(node);
}

void buffer_list::add_binarized_log_double_right (const log_double *v, int v_sz) {
  buffer_list_node *node = new buffer_list_node(v_sz*10);
  char *r = node->buffer;
  for (int i=0; i<v_sz; i++,r+=10)
    binarizer::code_double(v[i].log(),r);
  add_buffer_right(node);
}


