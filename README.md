april-ann
=========

Compilation
-----------

First, it is mandatory to configure (only the first time) the repo PATH and other stuff:

```$ . configure.sh```

Second, you could compile the april version which you need. We have developed compiling files for use
different libraries. It is simple, you do

```$ make TARGET```

where TARGET is one of the following, depending on which version you want:

- ATLAS: release/debug
- Intel MKL: release-mkl/debug-mkl
- Intel MKL + CUDA: release-cuda-mkl/debug-cuda-mkl
- Mac OS X Accelerate Framework: release-macosx/debug-macosx

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

Includes these sources
----------------------
- LUA virtual machine 5.1.4: http://www.lua.org/
- Luiz's lstrip for Lua 5.1: http://www.tecgraf.puc-rio.br/~lhf/ftp/lua/5.1/lstrip.tar.gz
- MersenneTwister: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
- GZIO: http://luaforge.net/projects/gzio/

Doxygen documentation
---------------------
- [Developer manual](http://pakozm.github.com/april-ann/doxygen_doc/developer/html/index.html)
- [Reference manual](http://pakozm.github.com/april-ann/doxygen_doc/user_refman/html/index.html)

MAC OS X
--------

- Install libpng
- Install findutils in /usr using --prefix=/usr
