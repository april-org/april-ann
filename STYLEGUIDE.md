# April-ANN source code **style guide**

The coding style is not homogeneous becase some code parts are very old. However,
for every new and fresh code, this guide will be followed.

## C/C++ style guide

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

#### Class declaration and implementation

- Separate the code in different files, in order to improve legibility
  and understanding. If one file requires to implement more than one
  class, there must be a good reason for it ;)

- Unless inline methods, it is better to separate declaration in .h
  and implementation in .cc or .cu (for C++ or Cuda).

#### Examples

```C++
// upper case underscored
#define ZERO_CONSTANT 0
namespace NameSpaceWhatever {
  // Two spaces indentation
  class ClassName : public OtherClassName {
    // upper case underscored
    const int ONE_CONSTANT = 1;
    // underscored property names
    int int_property;
    float float_property;
  public:
    ClassName(int int_property, float float_property) :
    OtherClassName(),
    int_property(int_property), float_property(float_property) {
    }
    // lowerCamelCase method names
    int getIntProperty() const { return int_property; }
    float getFloatProperty() const { return float_property; }
  };
  // lowerCamelCase function names
  template<typename T>
  T addTwo(const T &a, const T &b) {
    // underscored variables
    T auxiliary = a + b;
    return auxiliary;
  }
}
```

### Useful macros

- Unused variables: `#include "unused_variable.h"` and use `UNUSED_VARIABLE` macro.
- Ignored results: `#include "ignore_result.h"` and use `IGNORE_RESULT` macro.
- Asserts: `#include "april_assert.h"` and use `april_assert` macro.
- Error print and exit: `#include "error_print.h"` and use macros `ERROR_PRINT*` and `ERROR_EXIT*`.

## Lua style guide

Prefered always in lower case with underscore sepparating words.

## LuaPkg packages

Package names are prefered to be equal to the global Lua table where C/C++
classes and functions are binded, and where Lua code is written.
