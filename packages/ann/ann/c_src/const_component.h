/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2014, Francisco Zamora-Martinez
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
#ifndef CONSTCOMPONENT_H
#define CONSTCOMPONENT_H

#include "ann_component.h"
#include "disallow_class_methods.h"
#include "smart_ptr.h"

namespace ANN {

  class ConstANNComponent : public ANNComponent {
    APRIL_DISALLOW_COPY_AND_ASSIGN(ConstANNComponent);
    
  protected:
    AprilUtils::SharedPtr<ANNComponent> component;
    AprilUtils::LuaTable component_weights;
    
  public:
    ConstANNComponent(ANNComponent *component, const char *name);
    virtual ~ConstANNComponent();
    
    virtual Basics::Token *getInput() { return component->getInput(); }
    virtual Basics::Token *getOutput() { return component->getOutput(); }
    virtual Basics::Token *getErrorInput() { return component->getErrorInput(); }
    virtual Basics::Token *getErrorOutput() { return component->getErrorOutput(); }

    virtual void setInput(Basics::Token *tk) { component->setInput(tk); }
    virtual void setOutput(Basics::Token *tk) { component->setOutput(tk); }
    virtual void setErrorInput(Basics::Token *tk) { component->setErrorInput(tk); }
    virtual void setErrorOutput(Basics::Token *tk) { component->setErrorOutput(tk); }

    virtual void copyState(AprilUtils::LuaTable &dict) {
      component->copyState(dict);
    }

    virtual void setState(AprilUtils::LuaTable &dict) {
      component->setState(dict);
    }
    
    virtual Basics::Token *doForward(Basics::Token* input, bool during_training);
    
    virtual Basics::Token *doBackprop(Basics::Token *input_error);
    
    virtual void reset(unsigned int it=0);
    
    virtual ANNComponent *clone();

    virtual void build(unsigned int _input_size,
		       unsigned int _output_size,
		       AprilUtils::LuaTable &weights_dict,
		       AprilUtils::LuaTable &components_dict);

    virtual char *toLuaString();
  };
}

#endif // CONSTCOMPONENT_H_H
