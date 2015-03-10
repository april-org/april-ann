/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2015, Francisco Zamora-Martinez
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
#ifndef GIT_COMMIT
#define GIT_COMMIT UNKNOWN
#endif
#define STRINGFY(X) #X
#define TOSTRING(X) STRINGFY(X)
const char *__COMMIT_NUMBER__ = TOSTRING(GIT_COMMIT);
const char *__COMMIT_HASH__ = TOSTRING(GIT_HASH);
const char *__APRILANN_VERSION_MAJOR__ = TOSTRING(APRILANN_VERSION_MAJOR);
const char *__APRILANN_VERSION_MINOR__ = TOSTRING(APRILANN_VERSION_MINOR);
#undef STRINGFY
#undef TOSTRING
