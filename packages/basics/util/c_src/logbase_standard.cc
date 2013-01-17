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
#include "logbase_standard.h"

const float log_float::floatnearzero    = 1e-37;
const float log_float::rawscorenearzero = -99;
const float log_float::rawscorezero     = -1e12;
const float log_float::rawscoreone      = 0;

const double log_double::doublenearzero   = 1e-37;
const double log_double::rawscorenearzero = -85;
const double log_double::rawscorezero     = -1e8;
const double log_double::rawscoreone      = 0;

