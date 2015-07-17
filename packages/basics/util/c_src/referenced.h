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
#ifndef REFERENCED_H
#define REFERENCED_H

#define IncRef(x) (x)->incRef()
#define DecRef(x) if ((x)->decRef()) delete (x)
// to avoid deletion of objects before pointer return, releases the references
// but not deletes the object
#define ReleaseRef(x) (x)->decRef()

/**
 * @brief The class Referenced is used as base class for binded to Lua
 * classes. It implements reference counting interface.
 *
 * @note In general, all derived classes from Reference need to be @c IncRef
 * when they are allocated using @c new operator, and therefore @c DecRef
 * when the pointer is not longer needed.
 *
 * @see AprilUtils::SharedPtr class for an utility which allows to execute
 * @c IncRef and @c DecRef in an automatic way.
 */
class Referenced {
 private:
  int refs;    ///< Number of living references of this object.
 public:
  Referenced();
  virtual ~Referenced();
  virtual void incRef();
  virtual bool decRef();
  virtual int  getRef() const;
};

/**
 * The functions template AssignRef allows to change the reference of a pointer
 * taking into account the corresponding IncRef and DecRef.
 *
 * @param[out] dest It is the variable where to assign the value, its content
 * will be DecRef'ed.
 *
 * @param ref It is the variable with the value which you want to assign.
 */
template<typename T>
void AssignRef(T *&dest, T *ref) {
  if (dest != ref) {
    if (dest != 0) DecRef(dest);
    dest = ref;
    if (dest != 0) IncRef(dest);
  }
}

#endif // REFERENCED_H
