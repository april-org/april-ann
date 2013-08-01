/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013, Francisco Zamora-Martinez
 *
 * The APRIL-MLP toolkit is free software; you can redistribute it and/or modify it
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
#ifndef COMPLEX_NUMBER_H
#define COMPLEX_NUMBER_H
#define REAL_IDX 0
#define IMG_IDX  1
struct ComplexF {
  float data[2];
  ComplexF operator*(const ComplexF &other) const {
    ComplexF result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] +
			     this->data[IMG_IDX]*other.data[IMG_IDX]);
    result.data[IMG_IDX]  = (this->data[REAL_IDX]*other.data[IMG_IDX] +
			     this->data[IMG_IDX]*other.data[REAL_IDX]);
  }
  ComplexF operator+(const ComplexF &other) const {
    ComplexF result;
    result.data[REAL_IDX] = this->data[REAL_IDX]+other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]+other.data[IMG_IDX];
  }
};
struct ComplexD {
  double data[2];
  ComplexD operator*(const ComplexD &other) const {
    ComplexD result;
    result.data[REAL_IDX] = (this->data[REAL_IDX]*other.data[REAL_IDX] +
			     this->data[IMG_IDX]*other.data[IMG_IDX]);
    result.data[IMG_IDX]  = (this->data[REAL_IDX]*other.data[IMG_IDX] +
			     this->data[IMG_IDX]*other.data[REAL_IDX]);
  }
  ComplexD operator+(const ComplexD &other) const {
    ComplexD result;
    result.data[REAL_IDX] = this->data[REAL_IDX]+other.data[REAL_IDX];
    result.data[IMG_IDX]  = this->data[IMG_IDX]+other.data[IMG_IDX];
  }
};
#endif // COMPLEX_NUMBER_H
