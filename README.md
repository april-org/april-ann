APRIL-ANN
=========

[![Build Status](https://travis-ci.org/pakozm/april-ann.png?branch=master)](https://travis-ci.org/pakozm/april-ann) Travis CI (master branch)

[![Build Status](https://travis-ci.org/pakozm/april-ann.png?branch=devel)](https://travis-ci.org/pakozm/april-ann) Travis CI (devel branch)

User documentation
------------------

Follow [wiki](https://github.com/pakozm/april-ann/wiki) for user documentation.

Tutorials
---------

Tutorials are available at [april-ann-tutorials](https://github.com/pakozm/april-ann-tutorials) 
repository.

Contributions
-------------

Contributions are wellcome. Only pull requests to `devel` branch will be
accepted, so avoid to create pull requests to `master`. A Travis CI instance
will check that your request passes all tests. For Lua unit testing use the
package `basics/utest`, and for C++ unit testing the package `basics/gtest`.
At the end of this document there are information about Doxygen documentation
which can be useful for C/C++ developing. For Lua developing use the wiki.

It is possible to make a pull request directly to `master` branch for bug or
memory leak solving.

Dependencies and basic setup
----------------------------

Clone the repository and enter to it:

```
$ git clone https://github.com/pakozm/april-ann.git
$ cd april-ann
```

The first time you want to run APRIL-ANN, you need to install
dependencies in Linux (via apt-get) and in MacOS X (via MacPorts/Homebrew).
There is an automatic shellscript which covers majorly Ubuntu (other Debian
based distributions can be supported) and MacOS X systems. You just need
to run:

```$ ./DEPENDENCIES-INSTALLER.sh```

### Requirements, installed by previous script

Requires the following libraries. Versions are only orientative, it could work
with older and newer versions whenver the API was compatible.

- GNU C++ compiler (g++): v 4.7.2
- Only in Linux systems: Lua 5.2 headers to tell APRIL-ANN the default system
  path for Lua modules (`lua5.2-deb-multiarch.h` header).
- BLAS implementation: ATLAS (v. 3), Intel MKL (v. 10.3.6), MacOS Accelerate Framework
- Threads posix (pthread)
- Readline (libreadline)
- OpenMP
- LAPACK library, offered by liblapack_atlas, mkl_lapack, or MacOS Accelerate Framework
- LAPACKE library when compiling with ATLAS

The following libreries are recommended, but optional, you will need to remove
its package from the path `profile_build_scripts/package_list.lua`:

- [OPTIONAL] libpng: if you want to open PNG images, package `libpng`.
- [OPTIONAL] libtiff: if you want to open TIFF images, package `libtiff`.
- [OPTIONAL] libz: support for open of GZIPPED files, package `gzio`.
- [OPTIONAL] libzip: support for open ZIP packages, package `zip`.

For perform computation on GPU, this optional library, which has an specific
make target:

- [OPTIONAL] CUDA and CUBLAS: release 4.2.6.

Compilation
-----------

First, it is mandatory to configure the repo PATH and other stuff.
This step is needed every time you start a session, and before compilation:

```$ . configure.sh```

Second, you can compile the APRIL-ANN version which you need.
We have developed compiling files for using different libraries.
The most simple compilation way is

```$ make```

and it would detect automatically your platform (Darwin or Linux) and your
target SUFIX (mkl, atlas, macports or homebrew). The automatic targets are:

- **release** automatic platform and sufix detection target (the same as `make`
  without any argument).
- **debug** automatic platform and sufix detection target for debug version.
- **test** automatic platform and sufix detection target for test-debug version.

There are available more targets, just depending in your particular system
installation you can execute them by using `$ make TARGET`:

- **release-mkl** needs of MKL library installed at `/opt/MKL` as prefix.
- **release-atlas** needs of OMP and ATLAS library.
- **release-no-omp** needs ATLAS library.
- **release-cuda-mkl** needs CUDA and MKL installed at `/opt/MKL` as prefix.
- **release-macports** needs Mac OS X with MacPorts and Accelerate Framework.
- **release-homebrew** needs Mac OS X with Homebrew and Accelerate Framework.

Besides this targets, it is possible to compile for debug replacing release
string with **debug** string, and for testing replacing release by
**test-debug**.

The makefile has the following variables which can be forced by the user:

- **PREFIX** indicates the prefix for libraries and binaries. In Linux it is
  `/usr`, in Darwin it depends on MacPorts (`/opt/local`) or Homebrew
  (`/usr/local`).
- **LUALIB** indicates where Lua modules are installed, by default it is
  `$(PREFIX)/lib/lua/5.2`
- **BIN** indicates where you want to install binary files, by default it is
  `$(PREFIX)/bin`

Each of this targets will need a little configuration depending on your library
installation. For example, in order to compile with MKL, the file
`profile_build_scripts/build_mkl_release.lua` contains the following sections
(among others):

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

You need to especify the `-I` option to the compiler, and all the extra_libs
stuff related with MKL.  Exists one build file for each possible target:
`build_release.lua`, `build_debug.lua`, `build_mkl_release.lua`,
`build_mkl_debug.lua`, ... and so on.

The binary will be generated at `bin/april-ann`, which incorporates the Lua 5.2
interpreter and works without any dependency in Lua.  Besides, a shared library
will be generated at `lib/aprilann.so`, so it is possible to use `require` from
Lua to load APRIL-ANN in a standard Lua 5.2 interpreter.

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
APRIL-ANN v0.2.1-beta COMMIT 920  Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
Lua 5.2.2  Copyright (C) 1994-2013 Lua.org, PUC-Rio
> print "Hello World!"
Hello World!
```

- It is possible to use APRIL-ANN as a Lua module, loading only the packages
  which you need (i.e. `require("aprilann.matrix")`), or loading the full
  library (`require("aprilann")`). **Be careful**, the APRIL-ANN modules doesn't
  follow Lua guidelines and have lateral effects because of the declaration of
  tables, functions, and other values at the GLOBALs Lua table:

```
$ lua
Lua 5.2.2  Copyright (C) 1994-2013 Lua.org, PUC-Rio
> require "aprilann.matrix"
> require "aprilann"
APRIL-ANN v0.2.1-beta COMMIT 920  Copyright (C) 2012-2013 DSIC-UPV, CEU-UCH
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
  Author = {Francisco Zamora-Mart\'inez and Salvador Espa{\~n}a-Boquera and
	        Jorge Gorbe-Moya and Joan Pastor-Pellicer and Adri\'an Palacios-Corella},
  Note = {{https://github.com/pakozm/april-ann}},
  Title = {{APRIL-ANN toolkit, A Pattern Recognizer In Lua with Artificial Neural Networks}},
  Year = {2013}}
```

Publications
------------

List of research papers which uses this tool:


- Joan Pastor-Pellicer, Salvador España-Boquera, Francisco Zamora-Martínez,
  M. Zeshan Afzal, M.J. Castro-Bleda.
  [Insights on the Use of Convolutional Neural Networks for Document Image Binarization](http://link.springer.com/chapter/10.1007/978-3-319-19222-2_10),
  *IWANN*, *Advances in Computational Intelligence*, pages 115-126, 2015.

- Joan Pastor-Pellicer, Salvador España-Boquera, Francisco Zamora-Martínez,
  María José Castro-Bleda.
  [Handwriting Normalization by Zone Estimation using HMM/ANNs](http://people.sabanciuniv.edu/berrin/cs581/Papers/icfhr2014/data/4334a633.pdf),
  *ICFHR*, pages 633-638, 2014.

- Francisco Zamora-Martínez, Pablo Romeu, Paloma Botella-Rocamora, and Juan Pardo.
  [On-line learning of indoor temperature forecasting models towards energy efficiency](http://dx.doi.org/10.1016/j.enbuild.2014.04.034),
  *Energy and Buildings*, 83:162-172, 2014.

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
  [Some empirical evaluations of a temperature forecasting module based on Artificial Neural Networks for a domotic home environment](http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0004133502060211).
  In *IC3K - KDIR*, pages 206-211, 2012.

Our ancient ANN implementation in the former APRIL tookit was published here:

- S. España-Boquera, F. Zamora-Martinez, M.J. Castro-Bleda, J. Gorbe-Moya.
  [Efficient BP algorithms for general feedforward neural networks](http://link.springer.com/chapter/10.1007%2F978-3-540-73053-8_33#page-1).
  In *IWINAC*, pages 327-336, 2007.

Other projects using it
-----------------------

Competition participations and other projects where APRIL-ANN has been used:

- [Kaggle American Epilepsy Society Seizure Prediction Challenge](https://www.kaggle.com/c/seizure-prediction),
  system ESAI-CEU-UCH positioned as 4th in the leaderboard.
  [Download from GitHub](https://github.com/ESAI-CEU-UCH/kaggle-epilepsy) the code
  to run this system.

Packages
--------

APRIL-ANN is compiled following a package system. In the directory packages you could find a
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
- Mike Pall's advanced readline patch for Lua: http://smbolton.com/lua.html#readline
- Google C++ Testing Framework: https://code.google.com/p/googletest/
- (Included, but not used) Lua autocompletion rlcompleter release 2, by rthomas:
  https://github.com/rrthomas/lua-rlcompleter

Wiki documentation
------------------

- [PDF version](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.pdf)
- [HTML one-page](https://rawgithub.com/wiki/pakozm/april-ann/WIKI.html)


Doxygen documentation
---------------------

The documentation of the devel branch will be mantained as updated as possible
in the following links:

- [C/C++ developer manual](http://pakozm.hol.es/STUFF/doxygen_doc/developer/html/index.html)
- [C/C++ binding manual](http://pakozm.hol.es/STUFF/doxygen_doc/user_refman/html/index.html)

However, you can produce the Doxygen documentation of the branch where
you are working by using the makefile's `document` target. Please, note that
you need to have installed [Doxygen](www.doxygen.org) and
[Graphviz](http://www.graphviz.org/).

```
$ make document
$ open doxygen_doc/developer/html/index.html
```

The last command can be substituted by you opening the indicated
location in your prefered web browser ;)

LINUX dependencies installation
-------------------------------

Execute: `$ ./DEPENDENCIES-INSTALLER.sh`

If your distribution is not supported (currently only Ubuntu has support), then
install g++, libatlas-dev, libreadline-dev, libpng-dev, libtiff-dev, libz-dev,
libopenmp-dev, libzip-dev, liblua5.2-dev.

MAC OS X dependencies installation
----------------------------------

Via MacPorts:

- Install [MacPorts](http://www.macports.org/)
- Execute `$ ./DEPENDENCIES-INSTALLER.sh`

Or via HomeBrew:

- Install [Homebrew](http://brew.sh/)
- Execute `$ ./DEPENDENCIES-INSTALLER.sh`

Building new modules out of APRIL-ANN repository
------------------------------------------------

Currently this option is only available for Linux systems, despite it can be
done manually in MacOS X if you know how to. So, for Linux systems, you need
to install APRIL-ANN using the following commands (after you have downloaded
or cloned the main repository):

```
$ make
$ sudo make install
```

After, you need to link your software using the following commands:

```
$ g++ -fPIC -shared -o YOUR_MODULE_NAME.so *.o $(pkg-config --cflags --libs april-ann)
```

Once you have done this, you can load your module into APRIL-ANN using Lua
interpreter:

```
$ lua
> require "aprilann"
APRIL-ANN v0.4.0  Copyright (C) 2012-2015 DSIC-UPV, CEU-UCH
Compiled at Sat Jul 18 13:45:52 2015, timestamp 1437219952
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE.txt.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
> your_module = require "YOUR_MODULE_NAME"
```
