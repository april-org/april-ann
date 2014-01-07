//BIND_HEADER_H
#include "interest_points.h"
#include "utilImageFloat.h"
#include "bind_image.h"
#include "vector.h"
#include "pair.h"
#include <cmath>
#include <cctype>
using april_utils::Point2D;
using namespace InterestPoints;
//BIND_END



//BIND_FUNCTION interest_points.extract_points_from_image_old
{
  using april_utils::vector;

  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  vector<Point2D> *result=InterestPoints::extract_points_from_image_old(img);

  lua_createtable(L, result->size(), 0);
  for (unsigned int i=1; i <= result->size(); i++) {
    lua_createtable(L, 2, 0);
    lua_pushnumber(L, (*result)[i-1].x);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, (*result)[i-1].y);
    lua_rawseti(L, -2, 2);
    lua_rawseti(L, -2, i);
  }

  LUABIND_RETURN_FROM_STACK(-1); 
  delete result;
}
//BIND_END


//BIND_FUNCTION interest_points.extract_points_from_image
{
  using april_utils::vector;

  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  vector<Point2D> *local_minima = new vector<Point2D>();
  vector<Point2D> *local_maxima = new vector<Point2D>();

  InterestPoints::extract_points_from_image(img, local_maxima, local_minima);

  lua_createtable(L, local_minima->size(), 0);
  for (unsigned int i=1; i <= local_minima->size(); i++) {
    lua_createtable(L, 2, 0);
    lua_pushnumber(L, (*local_minima)[i-1].x);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, (*local_minima)[i-1].y);
    lua_rawseti(L, -2, 2);
    lua_rawseti(L, -2, i);
  }

//  LUABIND_RETURN_FROM_STACK(-1);
  
  lua_createtable(L, local_maxima->size(), 0);
  for (unsigned int i=1; i <= local_maxima->size(); i++) {
    lua_createtable(L, 2, 0);
    lua_pushnumber(L, (*local_maxima)[i-1].x);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, (*local_maxima)[i-1].y);
    lua_rawseti(L, -2, 2);
    lua_rawseti(L, -2, i);
  }

//  LUABIND_RETURN_FROM_STACK(-2); 
//  LUABIND_RETURN_FROM_STACK(-2);
  return 2; 
  delete local_minima;
  delete local_maxima;
}
//BIND_END

//-----------------------------------------
//
// CLASS INTEREST POINTS
//
//-----------------------------------------

//BIND_LUACLASSNAME SetPoints interest_points.SetPoints
//BIND_CPP_CLASS    SetPoints

//BIND_CONSTRUCTOR SetPoints
//DOC_BEGIN
//DOC_END
{
 using InterestPoints::SetPoints;
  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  SetPoints *obj = new SetPoints(img);
  LUABIND_RETURN(SetPoints, obj);
}
//BIND_END

//BIND_METHOD SetPoints getNumPoints
{
  LUABIND_RETURN(int, obj->getNumPoints());
}
//BIND_END
//BIND_METHOD SetPoints addPoint
{
 LUABIND_CHECK_ARGN(==, 2);
 LUABIND_CHECK_PARAMETER(1, int);
 LUABIND_CHECK_PARAMETER(2, table);

 // Recieves a table of 3-elements {x, y, c, type}
 // Sometimes can include a 5th element with the log-probability
 int elems;
 int comp = 1;
 LUABIND_GET_PARAMETER(1, int, comp);
 LUABIND_TABLE_GETN(1, elems);

 int x, y, c;
  
 bool type = true;
 float log_prob = 0.0;

 lua_rawgeti(L, -1, 1);
 x = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 2);
 y = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 3);
 c = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 4);
 type = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 bool class_type = (type == 0);
 if (elems > 4) {
   lua_rawgeti(L, -1, 4+c);
   log_prob = lua_tonumber(L,-1);
 }
 obj->addPoint(comp, x, y, c, class_type, log_prob);
}
//BIND_END


//BIND_METHOD SetPoints printComponents
{
  obj->print_components();
}
//BIND_END

//BIND_METHOD SetPoints sortByConfidence
{
  obj->sort_by_confidence();
}
//BIND_END

//BIND_METHOD SetPoints sortByX
{
  obj->sort_by_x();
}
//BIND_END

//BIND_METHOD SetPoints getSize
{
    LUABIND_RETURN(int, obj->getSize());
}
//BIND_END

//BIND_METHOD SetPoints getComponentPoints
{
    // Devuelve una lista de listas de tuplas ;)
    const vector <PointComponent> *v = obj->getComponents();
    int tupla_len = 3;         
    
    // Outer list
    lua_createtable (L, v->size(), 0);
    //fprintf(stderr, "GetComponent Size %d (%d) Points: %d\n", v->size(), obj->getSize(), obj->getNumPoints()); 
    for(int i=0; i < (int)v->size(); ++i) 
    {   
        int component_size = (*v)[i].size();

        //Inner list
        lua_createtable (L, component_size, 0);
        for (int j = 0; j < component_size; ++j) {
            // Tuple list
            lua_createtable (L, 3, 0);
            lua_pushint(L,(*v)[i][j].x);
            lua_rawseti(L,-2, 1);
            lua_pushint(L,(*v)[i][j].y);
            lua_rawseti(L,-2, 2);
            lua_pushint(L,(*v)[i][j].point_class);
            lua_rawseti(L,-2, 3);
            // Add the tuple to the outer list
            lua_rawseti(L, -2, j+1);
        }
        lua_rawseti(L,-2, i+1);
    }

    LUABIND_RETURN_FROM_STACK(-1);
}
//BIND_END


/*** Connected Points ***/

//BIND_LUACLASSNAME ConnectedPoints interest_points.ConnectedPoints
//BIND_CPP_CLASS ConnectedPoints
//BIND_SUBCLASS_OF ConnectedPoints SetPoints

//BIND_CONSTRUCTOR ConnectedPoints
//DOC_BEGIN
//DOC_END
{
  using InterestPoints::ConnectedPoints;
  LUABIND_CHECK_ARGN(==,1);
  ImageFloat *img;
  LUABIND_GET_PARAMETER(1, ImageFloat, img);

  ConnectedPoints *obj = new ConnectedPoints(img);
  LUABIND_RETURN(ConnectedPoints, obj);
}
//BIND_END

//BIND_METHOD ConnectedPoints addPoint
{
 LUABIND_CHECK_ARGN(==, 1);
 LUABIND_CHECK_PARAMETER(1, table);

 // Recieves a table of 3-elements {x, y, c, type}
 // Sometimes can include a 5th element with the log-probability
 int elems;

 LUABIND_TABLE_GETN(1, elems);

 int x, y, c;
  
 bool type = true;
 float log_prob = 0.0;

 lua_rawgeti(L, -1, 1);
 x = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 2);
 y = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 3);
 c = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 lua_rawgeti(L, -1, 4);
 type = (int)lua_tonumber(L,-1);
 lua_pop(L,1);

 bool class_type = (type == 0);
 if (elems > 4) {
   lua_rawgeti(L, -1, 4+c);
   log_prob = lua_tonumber(L,-1);
 }
 obj->addPoint(x, y, c, class_type, log_prob);
}
//BIND_END

//BIND_METHOD ConnectedPoints computeComponents
{
 LUABIND_CHECK_ARGN(==,0);

 SetPoints *sp = obj->computePoints();

 LUABIND_RETURN(SetPoints, sp);
}
//BIND_END

//BIND_METHOD SetPoints getLinearRegression
{
//Used for drawing the line regression

   lua_createtable(L, obj->getSize(), 0);
   int comp = 1;

   for (int i = 0; i < obj->getSize(); ++i) {
     PointComponent component = obj->getComponent(i);
     line *myLine = component.get_regression_line();
     if (!myLine){
        
         continue;
     }
     // Compute most left and most right points
     component.sort_by_x();
     interest_point p1 = component[0];
     interest_point p2 = component[component.size()-1];
     float dist = 0.0;
     Point2D cp1 = myLine->closestPoint(p1,dist);
     Point2D cp2 = myLine->closestPoint(p2,dist);
     //ini point and end point
     lua_createtable(L,2,0);

     lua_createtable(L,2,0);
     lua_pushint(L, (int)cp1.x);
     lua_rawseti(L, -2, 1);
     lua_pushint(L, (int)cp1.y);
     lua_rawseti(L, -2, 2);

     lua_rawseti(L,-2, 1);
     lua_createtable(L, 2,0);
     lua_pushint(L, (int)cp2.x);
     lua_rawseti(L, -2, 1);
     lua_pushint(L, (int)cp2.y);
     lua_rawseti(L, -2, 2);

     lua_rawseti(L, -2, 2);

     lua_rawseti(L, -2, comp);
     printf("Componente: %d, Punto A: (%d %d) (%f %f), Punto B: (%d, %d) (%f, %f)\n", i, p1.x, p1.y,cp1.x,cp1.y,  p2.x, p2.y, cp2.x, cp2.y);
     printf("Slope: %f, intercept %f\n", myLine->getSlope(), myLine->getYintercept());
     comp++;
   } 
   
   //Returns a table of 2 points
   LUABIND_RETURN_FROM_STACK(-1);

}
//BIND_END
