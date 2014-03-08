# April-ANN source code **style guide**

The coding style is not homogeneous becase some code parts are very old. However,
for every new and fresh code, this guide will be followed.

## C/C++ style guide

### Coding style

- Indentation: Two spaces indentation. For deep indentations, 7 spaces could be
  substitued by one tab.
- Namespaces: Namespaces are in UpperCamelCase.
- Classes: Class names (and structs) are in UpperCamelCase.
- Methods: Method names in lowerCamelCase.
- Properties: In lower case with underscore sepparating words.
- Variables: In lower case with underscore sepparating words.
- Constants: In upper case with underscore sepparating words.

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
