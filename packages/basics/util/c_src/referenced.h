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
#ifndef REFERENCED_H
#define REFERENCED_H

#define IncRef(x) x->incRef()
#define DecRef(x) if (x->decRef()) delete x
#define AssignRef(dest,ref) do {					\
      if ((dest)) DecRef((dest));					\
      (dest) = (ref);							\
      IncRef((ref));							\
  } while(0)

class Referenced {
 protected:
  int refs;
 public:
  Referenced();
  virtual ~Referenced();
  virtual void incRef();
  virtual bool decRef();
};

#endif // REFERENCED_H
