/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2013 Francisco Zamora-Martinez
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
#ifndef TOKEN_MANIP_H
#define TOKEN_MANIP_H

#include "token_base.h"

/// This function pushes in a bunch of patterns a new pattern. Patterns are
/// of type TokenMemoryBlock.
void pushTokenMemBlockAt(unsigned int bunch_pos, Token *bunch, Token *pat);

/// This function puhses in a bunch of patterns a new pattern. Patterns could
/// be any combination of TokenBunchVector's which internally contains
/// a TokenMemoryBlock.
void pushTokenAt(unsigned int bunch_pos, Token *&bunch, Token *pat);

#endif // TOKEN_MANIP_H
