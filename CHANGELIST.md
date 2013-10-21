ChangeList
==========

Master branch release
---------------------

- Added `matrix.svd` and `matrix.diagonalize`.
- Added `stats.pca`, `stats.mean_centered`, `stats.pca_whitening`.

v0.3.0-beta relase
------------------

### API Changes

- Added `loss` parameter to `trainable.supervised_trainer` methods.
- Added `optimizer` parameter to `trainable.supervised_trainer` methods.
- Added `ann.optimizer` package, which has the implementation of weight update
  based on weight gradient. So, the ANN components only compute gradients.
  This allows to implement different optimization methods (as "Conjugate
  Gradient", or "Linear Search Brack-Propagation") with the same gradient
  computation.
- Loss functions `ann.loss` API has been changed, now the loss computation is
  done in two separated steps:
      - `matrix = loss:compute_loss(output,target)`: which returns a matrix with
        the loss of every pair of patterns, allowing to perform several loss
		computations without taken them into account.
      - `matrix = loss:accum_loss(matrix)`: which accumulates the loss in the
	    internal state of the class.
- Added a new version of `loss` function, which computes mean and
  sample variance of the loss. Besides, the loss computation is done
  with doubles, being more accurated than before.
- `replacement` parameter in SDAE doesn't force `on_the_fly` parameter, they are
  independent.
- SDAE training has been changed in order to allow the use of LUA datasets,
- Replaced `cpp_class_binding_extension` by `class_extension` function,
  adding Lua classes support besides to CPP binded classes.
- Modified `class` and `class_instance` functions to be more homogeneous
  with C++ binding.
- Added support for GZipped matrices load and save from C++, so functions
  `matrix.savefile` and `matrix.loadfile` (and its correspondence for complex
  numbers, double, int32, and char) were removed. Methods `matrix.fromFilename`
  and `matrix.toFilename` accept '.gz' extension.

### Packages rename

- Changed package `sdae` to `ann.autoencoders`.
- Changed package `loss_functions` to `ann.loss`.
- Splitted `mathcore` package into `mathcore` and `complex` packages.
- Renamed `math` package to `mathcore` to avoid the collision with Lua standard
  math library.

### New features

- April-ANN is deployed as a standalone executable and as a shared library for
  Lua 5.2.
- Modified `lua.h` to incorporate the GIT commit number in the disclaimer.
- Added Lua autocompletion when readline is available.
- Implemented SignalHandler class in C++.
- Added `signal.register` and `signal.receive` functions to Lua.
- Added to `matrix` the methods `map`, `contiguous`, `join`, `abs`, `tan`,
  `atan`, `atanh`, `sinh`, `asin`, `asinh`, `cosh`, `acos`, `acosh`, `fromMMap`,
  `toMMap`, `div`, `max`, `min`.
- Added `iterator` class, which is a wrapper around Lua iterators, but
  provides a more natural interface with functional programming procedures
  as `map`, `filter`, `apply`, or `reduce`.
- Added methods `iterate`, `field`, `select` to iterator Lua class.
- `table.insert` returns the table, which is useful for reduction operations.
- Added `table` method to `iterator` class.
- Added naive `L1_norm` regularization.
- Added `dataset.clamp`.
- Added `mathcore.set_mmap_allocation` function, which allows to forces the
  use of mmap for `matrix` memory allocation.
- Added `ann.components.slice`.
- Added GS-PCA algorithm for efficient computation of PCA (iterative algorithm),
  `stats.iterative_pca` Lua function.
- Added basic MapReduce implementation in Lua.
- Added `stats.correlation.pearson` Lua class.
- Added `stats.bootstrap_resampling` function.
- Added `math.add`, `math.sub`, `math.mul`, `math.div` functions.
- `trainable` and `ann.mlp.all_all` are using `matrix:to_lua_string()`
  method.
- Added method `to_lua_string()` in all matrix types, so the method produce
  a Lua chunk which is loadable and produce a matrix.
- Added serialization to `parallel_foreach`, allowing to produce outputs which
  could be loaded by the caller process.
- Declaration of `luatype` function as global, it wasn't.
- Added `iterable_map` and `multiple_ipairs` functions to the Lua utilities.
- Added SubAndDivNormalizationDataSet, applies a substraction and a division of
  the feature vectors.
- Added stepDataset.

### Bugs removed

- Solved bug at `luabind_template.cc`, which introduced spurious segmentation
  faults due to Lua execution of garbage collection in the middle of a
  `lua_pushClassName`.
- Solved bug at glob function.
- Solved bug at matrix iterators operator=.
- Solved bug at method `matrix::best_span_iterator::setAtWindow`. Becaose of It
  the method didn't works when the matrix was a sub-matrix (slice) of other
  matrix.
- Solved bugs at Matrix template constructor which affects to `rewrap` lua
  method, and to select method, which affects to `select` lua method.
- Added binarizer::init() to a binded static_constructor, it is needed to
  execute init() before decode/encode double numbers, because of endianism.
- Solved bug at constString when extracting double numbers in binary format.
- MacOS compilation problems solved.
- Solved problems with CUDA, it is working again.
- Dynamic loading of C modules is working now.

### C/C++ code changes

- Added BIND_STRING_CONSTANT to luabind, so it is possible to export C string
  constants to Lua.
- Removed warning of clang about unused variables, adding a new macro
  `UNUSED_VARIABLE(x)` defined in the header `utils/c_src/unused_variable.h`.
- Matrix fromString and toString Lua methods have been improved to write/read
  directly from Lua string buffer, so the memory print has been reduced.
- The C++ routines to write and read files is generalized to work with streams,
  under the BufferedStream template, and it is instantiated to FILE and gzFile
  formats.
- Added sanity check to cross-entropy and multi-class cross-entropy loss
  functions, to detect the use of non logarithmic outputs.

v0.2.1-beta relase
------------------

- matrices with float (matrix), complex numbers (matrixComplex), double
  (matrixDouble), int32 (matrixInt32), and char (matrixChar).
- GZIO is fresh code, a wrapper over ZLIB, binded using LUABIND.
- Added LAPACK support, for matrix inversion computation.
- Updated Lua class mean_var to follow the method of Knuth.
- CUDA compiles, but stills not working :SSSSSS
- Solved problems of table.tostring.
- Generalization of matrix math operators.
- Specialization of math wrappers for float and complex numbers.
- Support for MAT-file format (Matlab matrices).
- Added mutual_information package.
- Added APRIL_EXEC and APRIL_TOOLS_DIR environment variables to configure.sh
- Added scripts for NBEST, but not working :SSSSSS
- Some memory leaks, and bugs are solved.
