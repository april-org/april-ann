ChangeList
==========

Master branch unstable release
------------------------------

- Added max and min methods over a given dimension for `matrix`.
- MacOS compilation problems solved.
- Matrix fromString and toString Lua methods have been improved to write/read
  directly from Lua string buffer, so the memory print has been reduced.
- The C++ routines to write and read files is generalized to work with streams,
  under the BufferedStream template, and it is instantiated to FILE and gzFile
  formats.
- Added sanity check to cross-entropy and multi-class cross-entropy loss
  functions, to detect the use of non logarithmic outputs.
- Solved problems with CUDA, it is working again.
- Dynamic loading of C modules is working now.
- Added support for GZipped matrices load and save from C++, so functions
  `matrix.savefile` and `matrix.loadfile` (and its correspondence for complex
  numbers, double, int32, and char) were removed. Methods `matrix.fromFilename`
  and `matrix.toFilename` accept '.gz' extension.

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
