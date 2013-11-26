April-ANN
=========

Requirements
------------

Requires the following libraries. Versions are only orientative, it could work
with older and newer versions whenver the API was compatible.

- GNU C++ compiler (g++): v 4.7.2
- Only in Linux systems: Lua 5.2 headers to tell April-ANN the default system
  path for Lua modules (`lua5.2-deb-multiarch.h` header).
- BLAS implementation: ATLAS (v. 3), Intel MKL (v. 10.3.6), MacOS Accelerate Framework
- Threads posix (pthread)
- Readline (libreadline)
- OpenMP
- LAPACK library, offered by liblapack_atlas, mkl_lapack, or MacOS Accelerate Framework
- libz: support for open of GZIPPED files

The following libreries are recommended, but optional:
- [OPTIONAL] libpng: if you want to open PNG images
- [OPTIONAL] libtiff: if you want to open TIFF images

For perform computation on GPU, this optional library:
- [OPTIONAL] CUDA and CUBLAS: release 4.2.6

Dependencies setup
------------------

The first time, you need to install dependencies in Linux (via apt-get) and in
MacOS X (via MacPorts) running:

```$ sudo ./DEPENDENCIES-INSTALLER.sh```

Compilation
-----------

First, it is mandatory to configure (only the first time) the repo PATH and other stuff:

```$ . configure.sh```

Second, you could compile the april version which you need. We have developed compiling files for using
different libraries. It is simple, you do

```$ make TARGET```

where TARGET is one of the following, depending on which version you want:

- ATLAS: `make release` (use build_release.lua), `make debug` (build_debug.lua)
- Intel MKL: `make` or `make release-mkl` (build_mkl_release.lua), `make debug-mkl` (build_mkl_debug.lua)
- Intel MKL + CUDA: `make release-cuda-mkl` (build_cuda_and_mkl_release.lua), `make debug-cuda-mkl` (build_cuda_and_mkl_debug.lua)
- Mac OS X Accelerate Framework: `make release-macosx` (build_release_macosx.lua), `make debug-macosx` (build_debug_macosx.lua)

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

The binary will be generated at `bin/april-ann`, which incorporates the Lua 5.2
interpreter and works without any dependency in Lua.  Besides, a shared library
will be generated at `lib/aprilann.so`, so it is possible to use `require` from
Lua to load April-ANN in a standard Lua 5.2 interpreter.

**NOTE** that loading `april-ann` as a Lua 5.2 module, you need to have the
`.so` library in the `package.cpath` or LUA_CPATH. It is possible to install it
in your system defaults following next section.

ENJOY!

Installation
------------

The installation is done executing:

```
$ sudo make install
```

This procedure copies the binary to system location in `/usr` (or in
`/opt/local` for Mac OS X via MacPorts). The shared library is copied
to Lua default directory, in order to load it by using `require` function.
*If you are using a non default installation (a custom one), please
copy the `.so` files manually to your `package.cpath` or LUA_CPATH*.

Use
---

- You can execute the standalone binary:

```
$ april-ann
April-ANN v0.2.1-beta COMMIT 920  Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
Lua 5.2.2  Copyright (C) 1994-2013 Lua.org, PUC-Rio
> print "Hello World!"
Hello World!
```

- It is possible to use April-ANN as a Lua module, loading only the packages
  which you need (i.e. `require("aprilann.matrix")`), or loading the full
  library (`require("aprilann")`). **Be careful**, the April-ANN modules doesn't
  follow Lua guidelines and have lateral effects because of the declaration of
  tables, functions, and other values at the GLOBALs Lua table:

```
$ lua
Lua 5.2.2  Copyright (C) 1994-2013 Lua.org, PUC-Rio
> require "aprilann.matrix"
> require "aprilann"
April-ANN v0.2.1-beta COMMIT 920  Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
> print "Hello World!"
Hello World!
```

Citation
--------

If you are interested in use this software, please cite correctly the source. In academic publications
you can use this bibitem:

```bibtex
@misc{aprilann,
  Author = {Francisco Zamora-Mart\'inez and Salvador Espa\~na-Boquera and
	        Jorge Gorbe-Moya and Joan Pastor-Pellicer and Adrian Palacios},
  Note = {{https://github.com/pakozm/april-ann}},
  Title = {{April-ANN toolkit, A Pattern Recognizer In Lua with Artificial Neural Networks}},
  Year = {2013}}
```

Publications
------------

List of research papers which uses this tool:

- Francisco Zamora-Martínez, Pablo Romeu, Paloma Botella-Rocamora, and Juan
  Pardo. [Towards Energy Efficiency: Forecasting Indoor Temperature via Multivariate Analysis](http://www.mdpi.com/1996-1073/6/9/4639).
  *Energies*, 6(9):4639-4659, 2013.

- Pablo Romeu, Francisco Zamora-Martinez, Paloma Botella, and Juan Pardo.
  [Time-Series Forecasting of Indoor Temperature Using Pre-trained Deep Neural Networks](http://dx.doi.org/10.1007/978-3-642-40728-4_57).
  In *ICANN*, pages 451-458. 2013.

- Joan Pastor-Pellicer, Francisco Zamora-Martinez, Salvador España-Boquera, and M.J. Castro-Bleda.
  [F-Measure as the error function to train Neural Networks](http://dx.doi.org/10.1007/978-3-642-38679-4).
  In *Advances in Computational Intelligence, IWANN, part I*, LNCS, pages 376-384. Springer, 2013.

- F. Zamora-Martínez, Pablo Romeu, Juan Pardo, and Daniel Tormo.
  Some empirical evaluations of a temperature forecasting module based on Artificial Neural Networks for a domotic home environment.
  In *IC3K - KDIR*, pages 206-211, 2012.

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
- Lua virtual machine 5.2.2: http://www.lua.org/
- Luiz's lstrip for Lua 5.1, adapted to compile with Lua 5.2.2: http://www.tecgraf.puc-rio.br/~lhf/ftp/lua/5.1/lstrip.tar.gz
- MersenneTwister: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
- Median filter from Simon Perreault: http://nomis80.org/ctmf.html
- RuningStat class for efficient and stable computation of mean and variance: http://www.johndcook.com/standard_deviation.html
- Lua autocompletion rlcompleter release 2, by rthomas: https://github.com/rrthomas/lua-rlcompleter

Wiki documentation
------------------

- [PDF version](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.pdf)
- [HTML one-page](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.html)


Doxygen documentation
---------------------
- [Developer manual](http://cafre.dsic.upv.es:8080/~pako/STUFF/doxygen_doc/developer/html/index.html)
- [Reference manual](http://cafre.dsic.upv.es:8080/~pako/STUFF/doxygen_doc/user_refman/html/index.html)

LINUX installation
------------------

Install g++, libatlas-dev, libreadline-dev, libpng-dev, libtiff-dev, libz-dev,
libopenmp-dev or execute: `$ sudo ./DEPENDENCIES-INSTALLER.sh`

MAC OS X installation
---------------------

Via MacPorts:

- Install [MacPorts](http://www.macports.org/)
- Execute `$ sudo ./DEPENDENCIES-INSTALLER.sh`

Or manually:

- Install libpng, from [sourceforge](http://sourceforge.net/projects/libpng/files/). Follow INSTALL information.
- Install findutils, from [GNU](http://ftp.gnu.org/pub/gnu/findutils/). Follow INSTALL instructions. Execute `./configure --prefix=/usr` in order to substitute BSD find of your MacOS.
