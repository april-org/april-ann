/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
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
#include "bind_dataset.h"
using namespace Basics;
//BIND_END

//BIND_HEADER_H
#include <errno.h>
#include <stdio.h>
#include "image_metrics.h"

using namespace Imaging;
//BIND_END


/////////////////////////////////////////////////////////////////////////////////

//BIND_LUACLASSNAME ImageMetrics image.image_metrics
//BIND_CPP_CLASS    ImageMetrics

//BIND_CONSTRUCTOR ImageMetrics
//DOC_BEGIN
// ImageMetrics()
// Creates an object for counter ant recover stadistics from comparision of images.
// White Pixels are representad as 1, and black as 0
//DOC_END
{
  ImageMetrics *obj = new ImageMetrics();
  LUABIND_RETURN(ImageMetrics, obj);
}
//BIND_END

//BIND_METHOD ImageMetrics process_dataset
//DOC_BEGIN
//Compute the comparition two datasets (extracted from images) to the metric class
//
//@param predicted dataset with the predicted values
//@param ground_truth dataset with the ground_truth values
//@param binary (optional) if enabled perform true/false comparision given a threshold
//@param threshold Necessary for binary classification
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  check_table_fields(L, 1, "predicted", "ground_truth", "binary", "threshold",
		     (const char *)0);
  DataSetFloat *ds, *GT;
  bool binary;
  float threshold;
  LUABIND_GET_TABLE_PARAMETER(1, predicted, DataSetFloat, ds);
  LUABIND_GET_TABLE_PARAMETER(1, ground_truth, DataSetFloat, GT);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, binary, bool, binary, false);
  LUABIND_GET_TABLE_OPTIONAL_PARAMETER(1, threshold, float, threshold, 0.5);
  obj->processDataset(ds,GT,binary,threshold);
}
//BIND_END

//BIND_METHOD ImageMetrics process_sample
//DOC_BEGIN
//compute only one sample
//@param pred predicted sample (i.e pixel)
//@param reference ground_truth sample
//DOC_END
{ 
  float pred, ref;
  LUABIND_CHECK_ARGN(==,2);
  LUABIND_GET_PARAMETER(1,float,pred);
  LUABIND_GET_PARAMETER(2,float,ref);
  obj->processSample(pred,ref);
}
//BIND_END

//BIND_METHOD ImageMetrics get_metrics
//DOC_BEGIN
//@returns a table with this information F-measure, Precission, Recall, GA, True Negative Rate and Accuracy
//DOC_END
{
  LUABIND_CHECK_ARGN(==,0);
  double FM,PR,RC,GA,TNR,ACC,MSE, PSNR, BRP, BRR, FNR;

  obj->getMetrics(FM,PR,RC,GA,MSE, TNR,ACC, PSNR, BRP, BRR, FNR);

  /*
  LUABIND_RETURN(float,FM);
  LUABIND_RETURN(float,PR);
  LUABIND_RETURN(float,RC);
  LUABIND_RETURN(float,GA);
  LUABIND_RETURN(float,MSE);
  LUABIND_RETURN(float,TNR);
  LUABIND_RETURN(float,ACC);
*/
  lua_newtable(L);

  lua_pushnumber(L,FM);
  lua_setfield(L,-2,"FM");

  lua_pushnumber(L,PR);
  lua_setfield(L,-2,"PR");

  lua_pushnumber(L,RC);
  lua_setfield(L,-2,"RC");

  lua_pushnumber(L,GA);
  lua_setfield(L,-2,"GA");

  lua_pushnumber(L,MSE);
  lua_setfield(L,-2,"MSE");

  lua_pushnumber(L,TNR);
  lua_setfield(L,-2,"TNR");

  lua_pushnumber(L,ACC);
  lua_setfield(L,-2,"ACC");
  
  lua_pushnumber(L, PSNR);

  lua_setfield(L,-2,"PSNR");
  lua_pushnumber(L,BRP);
  lua_setfield(L,-2,"BRP");

  lua_pushnumber(L,BRR);
  lua_setfield(L,-2,"BRT");

  lua_pushnumber(L,FNR);
  lua_setfield(L,-2,"FNR");
  LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END

//BIND_METHOD ImageMetrics combine
//DOC_BEGIN
//Add the information of the parameter Metrics to the metrics information
//@param ImageMetric1 Object with image metrics information to add to the actual instance
//DOC_END
{
  LUABIND_CHECK_ARGN(==,1);
  ImageMetrics *im1;
  LUABIND_GET_PARAMETER(1, ImageMetrics, im1);
  obj->combine(*im1);
}
//BIND_END

//BIND_METHOD ImageMetrics clone
{
  LUABIND_CHECK_ARGN(==,0);
  LUABIND_RETURN(ImageMetrics,obj->clone());
}
//BIND_END


//BIND_METHOD ImageMetrics clear
{
  LUABIND_CHECK_ARGN(==,0);
  obj->clear();
}
//BIND_END

//BIND_METHOD ImageMetrics nSamples
{
  LUABIND_CHECK_ARGN(==,0);
  int nSamples = obj->nSamples();
  LUABIND_RETURN(int,nSamples);
}
//BIND_END

//////////////////////////////////////////////////////////////////////

