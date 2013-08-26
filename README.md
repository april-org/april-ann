April-ANN
=========

Requirements
------------

Requires the following libraries. Versions are only orientative, it could work with older and newer versions.

- GNU C++ compiler (g++): v 4.7.2
- BLAS implementation: ATLAS (v. 3), Intel MKL (v. 10.3.6), MacOS Accelerate Framework
- Threads posix (pthread)
- Readline (libreadline)
- OpenMP
- LAPACK library, offered by liblapack_atlas, mkl_lapack, or MacOS Accelerate Framework

The following libreries are recommended, but optional:
- [OPTIONAL] libpng: if you want to open PNG images
- [OPTIONAL] libtiff: if you want to open TIFF images
- [OPTIONAL] libz: if you want to open GZIPPED files

For perform computation on GPU, this optional library:
- [OPTIONAL] CUDA and CUBLAS: release 4.2.6

Compilation
-----------

First, it is mandatory to configure (only the first time) the repo PATH and other stuff:

```$ . configure.sh```

Second, you could compile the april version which you need. We have developed compiling files for using
different libraries. It is simple, you do

```$ make TARGET```

where TARGET is one of the following, depending on which version you want:

- ATLAS: make release (use build_release.lua), make debug (build_debug.lua)
- Intel MKL: make or make release-mkl (build_mkl_release.lua), make debug-mkl (build_mkl_debug.lua)
- Intel MKL + CUDA: make release-cuda-mkl (build_cuda_and_mkl_release.lua), make debug-cuda-mkl (build_cuda_and_mkl_debug.lua)
- Mac OS X Accelerate Framework: make release-macosx (build_release_macosx.lua), make debug-macosx (build_debug_macosx.lua)

Each of this targets will need a little configuration depending on your library
installation. For example, in order to compile with MKL, the file build_mkl_release.lua contains
the following sections (among others):

```
  global_flags = {
    debug="no",
    use_lstrip = "no",
    use_readline="yes",
    optimization = "yes",
    platform = "unix",
    extra_flags={
      -- For Intel MKL :)
      "-DUSE_MKL",
      "-I/opt/MKL/include",
      --------------------
      "-march=native",
      "-msse",
      "-DNDEBUG",
    },
    extra_libs={
      "-lpthread",
      -- For Intel MKL :)
      "-L/opt/MKL/lib",
      "-lmkl_intel_lp64",
      "-Wl,--start-group",
      "-lmkl_intel_thread",
      "-lmkl_core",
      "-Wl,--end-group",
      "-liomp5"
    },
  },
```

You need to especify the `-I` option to the compiler, and all the extra_libs stuff related with MKL.
Exists one build file for each possible target: build_release.lua, build_debug.lua, build_mkl_release.lua,
build_mkl_debug.lua, ... and so on.

ENJOY!

Citation
--------

If you are interested in use this software, please cite correctly the source. In academic publications
you can use this bibitem:

```bibtex
@misc{aprilann,
        Author = {Francisco Zamora-Mart\'inez and Salvador Espa\~na-Boquera and Jorge Gorbe-Moya and Joan Pastor-Pellicer and Adrian Palacios},
        Note = {{https://github.com/pakozm/april-ann}},
        Title = {{April-ANN toolkit, A Pattern Recognizer In Lua with Artificial Neural Networks}},
        Year = {2013}}
```


Packages
--------

April-ANN is compiled following a package system. In the directory packages you could find a
tree of directory entries. Leaves in the tree are directories which contain file "package.lua".
The "package.lua" defines requirements, dependencies, libraries, and other stuff needed by the
corresponding package.

Each package could contain this directories:

- c_src: source files (.h, .cc, .c, .cpp, .cu, and others).
- binding: binding files (.lua.cc), a kind of templatized file which generates the glue code between C/C++ and Lua.
- lua_src: lua source files which define functions, tables, and pseudo-classes in Lua.
- doc: doxygen documentation additional files.
- test: examples and files for testing.

At root directory exists a file named "package_list.lua". It is a Lua table with the name of packages that
you want to compile. If you don't want or don't have libpng, or libtiff, or other library, you could
erase the package name from this list to avoid its compilation.


Includes these sources
----------------------
- LUA virtual machine 5.2.2: http://www.lua.org/
- Luiz's lstrip for Lua 5.1, adapted to compile with Lua 5.2.2: http://www.tecgraf.puc-rio.br/~lhf/ftp/lua/5.1/lstrip.tar.gz
- MersenneTwister: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
- Median filter from Simon Perreault: http://nomis80.org/ctmf.html
- RuningStat class for efficient and stable computation of mean and variance: http://www.johndcook.com/standard_deviation.html


Wiki documentation
------------------

- [PDF version](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.pdf)
- [HTML one-page](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.html)


Doxygen documentation
---------------------
- [Developer manual](http://pakozm.github.com/april-ann/doxygen_doc/developer/html/index.html)
- [Reference manual](http://pakozm.github.com/april-ann/doxygen_doc/user_refman/html/index.html)

LINUX installation
------------------

Install g++, libatlas-dev, libreadline-dev, libpng-dev, libtiff-dev, libz-dev, libopenmp-dev.

MAC OS X installation
---------------------

- Install libpng, from (sourceforge)[http://sourceforge.net/projects/libpng/files/]. Follow INSTALL information.
- Install findutils, from (GNU)[http://ftp.gnu.org/pub/gnu/findutils/]. Follow INSTALL instructions. Execute `./configure --prefix=/usr` in order to substitute BSD find of your MacOS.
