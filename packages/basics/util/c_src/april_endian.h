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
#ifndef APRIL_ENDIAN_H
#define APRIL_ENDIAN_H

// Endianness checks
// TODO: these ones are possibly unsafe, we should execute
// a test before compiling and define endianness accordingly.
// (like configure does in a typical build-system)

#define APRIL_BIG_ENDIAN 0xdeadbeef
#define APRIL_LITTLE_ENDIAN 0xefbeadde

#ifndef APRIL_ENDIANNESS
#if defined(__POWERPC__)|| defined(__ppc__) || defined(_M_PPC) || \
    defined(_M_M68K) || defined(__m68k__) || defined(mc68000) || \
    (defined(__MIPS__) && defined(__MISPEB__)) || \
    defined(__hppa__) || \
    defined(__sparc__)
#define APRIL_ENDIANNESS APRIL_BIG_ENDIAN
#else
#define APRIL_ENDIANNESS APRIL_LITTLE_ENDIAN
#endif
#endif // !defined(APRIL_ENDIANNESS)

#endif // !defined(APRIL_ENDIAN_H)
