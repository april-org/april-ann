/*
 * This file is part of the Neural Network modules of the APRIL toolkit (A
 * Pattern Recognizer In Lua).
 *
 * Copyright 2012, Salvador España-Boquera, Adrian Palacios Corella, Francisco
 * Zamora-Martinez
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
//BIND_HEADER_C
//BIND_END

//BIND_HEADER_H
#include "activation_function.h"
using namespace ANN;
//BIND_END

//BIND_LUACLASSNAME ActivationFunction         ann.__activation_function__
//BIND_LUACLASSNAME LogisticActivationFunction ann.activations.logistic
//BIND_LUACLASSNAME TanhActivationFunction     ann.activations.tanh
//BIND_LUACLASSNAME SoftmaxActivationFunction  ann.activations.softmax

//BIND_CPP_CLASS ActivationFunction
//BIND_CPP_CLASS LogisticActivationFunction
//BIND_CPP_CLASS TanhActivationFunction
//BIND_CPP_CLASS SoftmaxActivationFunction

//BIND_SUBCLASS_OF LogisticActivationFunction ActivationFunction
//BIND_SUBCLASS_OF TanhActivationFunction     ActivationFunction
//BIND_SUBCLASS_OF SoftmaxActivationFunction  ActivationFunction

//BIND_CONSTRUCTOR ActivationFunction
//DOC_BEGIN
// __activation_function__()
/// Forbidden constructor, is an abstract class.
//DOC_END
{
  LUABIND_ERROR("Abstract class!!!\n");
}
//BIND_END

//BIND_CONSTRUCTOR LogisticActivationFunction
//DOC_BEGIN
// object logistic()
/// Logistic activation function constructor. Builds a C++ object exported to
/// LUA that could be used in any ANN as an activation function. It follows this
/// equation, for a given neuron potential y:
/// \f[
/// f(y) = \frac{1}{1 + e^{-y}}
/// \f]
//DOC_END
{
  obj = new LogisticActivationFunction();
  LUABIND_RETURN(LogisticActivationFunction, obj);
}
//BIND_END

//BIND_CONSTRUCTOR TanhActivationFunction
//DOC_BEGIN
// object tanh()
/// Tanh activation function constructor. Builds a C++ object exported to LUA
/// that could be used in any ANN as an activation function. It follows this
/// equation, for a given neuron potential y:
/// \f[
/// f(y) = \frac{2}{1 + e^{-y}} - 1
/// \f]
//DOC_END
{
  obj = new TanhActivationFunction();
  LUABIND_RETURN(TanhActivationFunction, obj);
}
//BIND_END

//BIND_CONSTRUCTOR SoftmaxActivationFunction
//DOC_BEGIN
// object softmax()
/// Softmax activation function constructor. Builds a C++ object exported to LUA
/// that could be used in any ANN as an activation function. It follows this
/// equation, for a given set of neuron potentials y:
/// \f[
/// a_i = \frac{e^{y_i}}{\sum_{j} e^{y_j}}
/// \f]
//DOC_END
{
  obj = new SoftmaxActivationFunction();
  LUABIND_RETURN(SoftmaxActivationFunction, obj);
}
//BIND_END
