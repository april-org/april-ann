ChangeList
==========

Master branch release
---------------------

### Unstable changes

- Added `Matrix<T>::convolution` method and `matrix:convolution` lua
  method. They are in experimental stage, please, be advice to use them by your
  own risk.
- Because of the changes in `Image`, several classes in package `imaging` has
  been reviewed, consistent tests are necessary to merge with master branch.

### API Changes

- Added new `class` behavior taken from
  [Lua OOP-iter](https://github.com/pakozm/lua-oop-iter), in order to introduce
  more modularity in APRIL-ANN.
- New `april_doc` and `april_set_doc` API, works with values instead of strings.
- Added `table.values`, `table.ivalues`, `table.keys` and `table.ikeys`
  iterators.

### Bugs removed

- Solved bug in `constString::extract_float` and `constString::extract_double`
  methods.

### C/C++

- Added `basics` namespace which stores almost all C/C++ code in `basics`
  packages.
- Simplified read/write of matrices using the new `StreamInterface` objects.
- Added new `StreamInterface` for input/output operations, with specializations
  for C files, C strings, Lua strings, GZIP files, ZIP files. Other
  specializations are easier to implement in the future.
- `Image` class code has been reviewed to be more coherent with current `Matrix`
  class implementation.

### Other

v0.3.1
------

### Unstable changes

- Added `clustering` and `knn` algorithms.
- Added `ann.components.zca_whitening` and `ann.components.pca_whitening`
  classes.
- Added `stats.zca_whitening` and `stats.pca_whitening` functions.
- Added packages `autodiff` and `autodiff.ann`, with automatic differentiation
  algorithms.
- Added `ann.optimizer.quickprop` algorithm, but stills untested.
- Gradients smoothing, based on weights sharing and `bunch_size`, is applied by
  `trainable.supervised_trainer` objects, not by the `optimizer`.
- Added `ann.components.dropout` class.
- Added `ann.autoencoders.ae` class, in order to factorize the SDAE code.
- Solved problems with CUDA compilation, but it stills not working because
  an error during cuMemAlloc.
- Class interest_points and layout algorithms (still in development).
- Added the geometry c++ class for working in to lines and point geometry.

### API Changes

- Modified bootstrapping API in order to be similar with R: `stats.boot()` and
  `stats.boot.ci()`.
- Added `ann.optimizer.asgd` for Averaged SGD optimization.
- Added `trainable.qlearning_trainer` for reinforcement learning tasks.
- Enhanced `util.serialize` and `util.deserialize` to accept functions as
  argument, allowing to serialize/deserialize over general streams.
- Added `iscallable(...)` function.
- Added `utest` package, for unit testing.
- Added methods to `matrix` which work with sparse matrices.
- Added class `matrix.sparse`.
- Added method `m:toTabStream(file or gzfile)` to `matrix`.
- Added operations `math.log1p` and `math.logadd`.
- Added operator `%` to Lua `string` metatable, allowing to do Python-like
  string formatting, and map-like substitutions.
- Added index matrix for min/max operations in `matrix` objects.
- Added `serialize` and `deserialize` Lua functions, in `util` package.

### Bugs removed

- Solved bug which makes to load unitialized weight matrices when loading old
  trainable.supervised_trainer (without `matrix.dict` object).
- Solved bug at `Matrix<T>::sliding_window` which makes an early end of the
  iterator when window has an `offset` different of 0.

### C/C++

- Added `-lcblas` option to `build_release.lua` and `build_debug.lua`.
- Added sparse CBLAS wrappers, for CUDA (not tested) and standard release.
- Added CBLAS wrappers for float, double, and ComplexF types.
- Added geometric parametrizer for off-line text preprocessing.
- Added dependency with `liblapacke` for PCA when not available MKL or MacOS X.

### Other

- Added argument check at `mathcore.block.*` constructors.
- Travis is monitoring only master and devel branches, for push and
  pull-requests.
- Added travis compilation and testing platform for pull-requests.

v0.3.1-alpha (pre-relase of v0.3.1-beta)
----------------------------------------

### API Changes

- Added `matrix:linspace(a,b)` and `matrix:linspace(a,b,base)`.
- `matrix:transpose()` returns a matrix which references the original. Any
  modification to the tranposed matrix, will be reflected at the original.
- `matrix:cmul` method is now **in-place**, so, if you don't want to modify the
  original matrix, you need to do `m:clone():cmul(...)` instead of
  `m:cmul(...)`.
- `update` property is added to `ann.optimizers.sgd` object in order to compute
  the momentum.
- Optimizers `execute` method receives a functions which computes the loss, the
  gradient (mandatory). Any other data is **optional**.
- Removed dropout code from activation functions.
- Deleted options from ANN components.
- Added methods `unroll` and `get` to `ann.components.stack` class.
- Added `inf` and `sup` limits to Hard-Tanh activation function.
- Added `random:to_lua_string()` method.
- Moved `ann.loss.__base__` to `ann.loss`.
- Moved `ann.components.actf.__base__` to `ann.components.actf`.

### New features

- Added Sauvola's binarization method.
- Added a normalize handwritting text utilities based on the main areas of the text.
- Added `matrix.dict`, a hash set dictionary in C++ binded to Lua, which allows
  to execute basic math operations and reductions over the whole set of
  contained matrices. It major purpose is to represent a set of connection
  weights or gradients in ANNs.
- Added `dataset.token.filter`, which allows ANN components as filters.
- Added `trainable.train_holdout_validation` class, which replace
  `trainable.supervised_trainer:train_holdout_validation` method.
- Added `trainable.train_wo_validation` class, which replace
  `trainable.supervised_trainer:train_wo_validation` method.
- Added `trainable.dataset_pair_iterator` and
  `trainable.dataset_multiple_iterator`, useful to iterate over datasets
  following different traversal schemes: sequential, shuffled, shuffled with
  replacement, shuffled with distribution.
- Added method `precompute_output_size` in ANN components.
- Added `ann.optimizer.cg`, Conjugate Gradient algorithm.
- Added `ann.optimizer.rprop`, Resilient Prop algorithm.
- Added `batch_fmeasure_micro_avg` and `batch_fmeasure_macro_avg` for
  multi-class FMeasure computation.
- Renamed loss function `local_fmeasure` as `batch_fmeasure`, and improved to
  work with multi-class models.
- Added `ann.loss.zero_one` loss function.
- Added `DEPENDENCIES-INSTALLER.sh`.
- Added syntactic sugar for `matrix:slice(...)` method: `m("1:2","3:4")` or
  `m({1,2},{3,4})`, like in Matlab or Octave.
- Added `matrix.svd` and `matrix.diagonalize`.
- Added `stats.pca`, `stats.mean_centered`, `stats.pca_whitening`.

### Bugs removed

- Memory leak due to the GPUMirroredMemoryBlock pool was solved.
- Solved bug at `stats.correlation.pearson`.
- Solved bug at `trainable` when using `use_dataset`, `train_dataset`,
  `validate_dataset`, `grad_check_dataset` methods without a `bunch_size`
  parameter, and with a trainer which has not a `bunch_size` defined at
  construction.
- Stabilization of log-logistic activation function.
- Stabilization of training with cross-entropy and multi-class-entropy.
- Solved bug when reading using `matrix.fromTabFilename`. The loader failed
  when the file had empty lines.
- Solved bug at `Matrix<T>::select(...)` C++ method. The matrix offset wasn't be
  added to the resulting matrix offset.
- Solved bug at `SlidingWindow::setAtWindow(...)` C++ method. The matrix offset
  wasn't be added to the computed window position.
- Solved bug at `buffered_memory.h`. Such bug introduces an early stop when
  reading matrices, ignoring the last lines of files.
- Solved problem with `rlcompleter`, which didn't work properly when loaded as a
  Lua module.
- Modified `configure.sh` to inform about any error during Lua building.
- Loadable modules are working on MacOs X.

### C/C++ code changes

- Added `Matrix<T>::random_acces_iterator`, which reduces the access overhead
  for random access of a `Matrix` object. It retains the memory pointer forcing
  an update between host and device (GPU) memory.
- Generalized `GPUMirroredBlockBase` to allow the reinterpretation of the
  underlying memory pointer using different types (reinterpret_cast).
- Added `MatrixSet` class template, which stores a dictionary of STRING->MATRIX,
  useful for ANNs and gradient descent purposes.
- Added `StochasticANNComponent` which is base class for stochastic components.
- Simplified coupling between ANN components, ANN loss functions, introducing
  automatic binding between MatrixFloat and TokenMatrixFloat.
- ANN Components has a pointer to a `MatrixFloat` instead of `ANN::Connections`.
- `ANN::Connections` is a static class with helper functions, and it is binded
  as `ann.connections`.
- Old-weights property is removed from ANN connections.
- Added automatic conversion between DataSetFloat and DataSetToken in
  `dataset.token.filter` and `dataset.token.union`.
- Added `FunctionInterface` class, in Lua as `functions` class, superclass of
  ANN components.

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

- APRIL-ANN is deployed as a standalone executable and as a shared library for
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
