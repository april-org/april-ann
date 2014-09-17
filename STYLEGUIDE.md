# APRIL-ANN source code **style guide**

The coding style is not homogeneous becase some code parts are very old. However,
for every new and fresh code, this guide will be followed.

## C/C++ style guide

Requirements and sugestions:

- Use always a namespace where new code will be declared. In this way, the
  developer documentation will be clearer.
- Put public methods and properties in the top of class definitions, then
  protected stuff and finally private stuff. This allows a better reading of
  interfaces directly from code.
- Use doxygen documentation style like in the example above.
- For large code blocks (between { ... }) it is nice to indicate at the closing
  brace which code is being closed, using a simple comment with //.
- **Smart pointers** are being introduced recently. The can be useful to be used
  in function arguments and in classes properties. In this way, becomes
  idiomatic the meaning of how the function or the class is being to use the
  given reference.

### Coding style

#### General rules

- Indentation: Two spaces indentation. Tabs are not recommended, please,
  substitute them by spaces.
- Namespaces: Namespaces are in UpperCamelCase.
- Classes: Class names (and structs) are in UpperCamelCase.
- Methods: Method names in lowerCamelCase.
- Properties: In lower case with underscore sepparating words.
- Variables: In lower case with underscore sepparating words.
- Constants: In upper case with underscore sepparating words.
- Virtual methods: Always let the `virtual` clause in derived classes, in order
  to allow the code to indicate it is polymorphic.

#### Class declaration and implementation

- Separate the code in different files, in order to improve legibility
  and understanding. If one file requires to implement more than one
  class, there must be a good reason for it ;)

- Unless inline methods, it is better to separate declaration in .h
  and implementation in .cc or .cu (for C++ or Cuda).

- In derived classes write always **virtual** in rewritten methods
  and *destructors*.

#### Examples

```C++
/// A briefly documented stuff, use this style.
#define ZERO_CONSTANT 0
/// My namespace whatever.
namespace NameSpaceWhatever {
  /**
   * @brief Brief documentation for this class.
   *
   * Use this style for more complex documentation, allowing to introduce @brief,
   * @param, @return, @note, @see, among other Doxygen tags.
   */
  class ClassName : public OtherClassName {
  public:
    /// Constructor.
    ClassName(int int_property, float float_property) :
    OtherClassName(),
    int_property(int_property), float_property(float_property) {
    }
    /// Getter.
    int getIntProperty() const { return int_property; }
    /// Getter.
    float getFloatProperty() const { return float_property; }
  private:
    /// An int constant.
    const int ONE_CONSTANT = 1;
    /// An int property.
    int int_property;
    /// A float property.
    float float_property;
  }; // class ClassName
  /**
   * @brief A function which adds two values.
   *
   * This function adds two values using the awesome add operator.
   *
   * @param a - The first value passed by reference.
   *
   * @param b - The second value passed by reference.
   */
  template<typename T>
  T addTwo(const T &a, const T &b) {
    // underscored variables
    T auxiliary = a + b;
    return auxiliary;
  }
} // namespace NameSpaceWhatever
```

Class derivation:

**parent.h**

```C++
/*
 *LICENSE COMMENT
 */
#ifndef PARENT_H
#define PARENT_H
class Parent {
public:
  Parent();
  virtual ~Parent();
  virtual void method();
};
#endif // PARENT_H
```

**parent.cc**
```C++
#include "parent.h"

Parent::Parent() {
  ...
}

Parent::~Parent() {
  ...
}

void Parent::method() {
  // Parent class method implementation
}
```

**derived.h**
```C++
/*
 *LICENSE COMMENT
 */
#ifndef DERIVED_H
#define DERIVED_H
#include "parent.h"

class Derived : public Parent {
public:
  Derived();
  virtual ~Derived();
  virtual void method();
};
#endif // DERIVED_H
```

**derived.cc**
```C++
#include "derived.h"

Derived::Derived() {
  ...
}

Derived::~Derived() {
  ...
}

void Derived::method() {
  // Derived class method implementation
}
```

### Useful macros

- Unused variables: `#include "unused_variable.h"` and use `UNUSED_VARIABLE`
  macro.
- Ignored results: `#include "ignore_result.h"` and use `IGNORE_RESULT` macro.
- Asserts: `#include "april_assert.h"` and use `april_assert` macro.
- Error print and exit: `#include "error_print.h"` and use macros `ERROR_PRINT*`
  and `ERROR_EXIT*`.
- Disallow copy constructor and assignment operator: `#include
  "disallow_class_methods.h"` and use `APRIL_DISALLOW_ASSIGN(type)` or
  `APRIL_DISALLOW_COPY_AND_ASSIGN(type)` in the private section of your class.

## Lua style guide

Prefered always in lower case with underscore sepparating words.

## LuaPkg packages

Package names are prefered to be equal to the global Lua table where C/C++
classes and functions are binded, and where Lua code is written.
