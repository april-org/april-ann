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
#ifndef COMPONENT_PROPERTIES_H
#define COMPONENT_PROPERTIES_H

// Properties for VirtualMatrixANNComponent and MatrixInputSwitchANNComponent
#define DEFAULT_INPUT_CONTIGUOUS_PROPERTY false

namespace ANN {

  class ComponentPropertiesAndAsserts {
  private:
    
    // Properties and asserts of input matrices needed by derived classes, and
    // ensured or asserted by this class. CURRENTLY ONLY ONE PROPERTY
    
    /// Contiguous property for input matrices (input and error_input)
    bool input_contiguous;
    
  protected:
    
    // Properties and asserts setters. It is recommended to call always this
    // setters in order to make clear the properties of your class.
    void setInputContiguousProperty(bool v) { input_contiguous = v; }
    
    // Properties and asserts getters.
    bool getInputContiguousProperty() const { return input_contiguous; }
    
    /////////////////////////////////////////////////////////////////////////
    
    ComponentPropertiesAndAsserts() :
      input_contiguous(DEFAULT_INPUT_CONTIGUOUS_PROPERTY) {
    }
    virtual ~ComponentPropertiesAndAsserts() {
    }
  };

}
#endif // COMPONENT_PROPERTIES_H
