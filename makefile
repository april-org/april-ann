LINUX_LUALIB = /usr/lib/lua/5.2
DARWIN_LUALIB = /opt/local/lib/lua/5.2
BIN = /usr/bin
UNAME = `uname`

ALL: release

release: release-mkl

debug: debug-no-omp

test: test-debug-atlas

document:
	lua profile_build_scripts/build_release_atlas.lua document

#############################################################################

# TEST with OMP and ATLAS
test-debug-atlas: debug-atlas
	lua profile_build_scripts/build_debug_atlas.lua test

# TEST without OMP and ATLAS
test-debug-no-omp: debug-no-omp
	lua profile_build_scripts/build_debug_no_omp.lua test

# TEST for MACOSX MACPORTS
test-debug-macports: debug-macports
	lua profile_build_scripts/build_debug_macports.lua test

# TEST for MACOSX HOMEBREW
test-debug-homebrew: debug-homebrew
	lua profile_build_scripts/build_debug_homebrew.lua test

# TEST with OMP and MKL
test-debug-mkl: debug-mkl
	lua profile_build_scripts/build_debug_mkl.lua test

# TEST with CUDA and MKL
test-debug-cuda-mkl: debug-cuda-mkl
	lua profile_build_scripts/build_debug_cuda_and_mkl.lua test

# TEST for raspberry-pi (subset of packages)
test-debug-pi: debug-pi
	lua profile_build_scripts/build_debug_pi.lua test

#############################################################################

# RELEASE for MACOSX MACPORTS
release-macports:
	lua profile_build_scripts/build_release_macports.lua

# RELEASE for MACOSX HOMEBREW
release-homebrew:
	lua profile_build_scripts/build_release_homebrew.lua

# RELEASE with OMP and MKL
release-mkl:
	lua profile_build_scripts/build_release_mkl.lua

# RELEASE with OMP and ATLAS
release-atlas:
	lua profile_build_scripts/build_release_atlas.lua

# RELEASE with CUDA and MKL
release-cuda-mkl:
	lua profile_build_scripts/build_release_cuda_and_mkl.lua

# RELEASE for raspberry-pi (subset of pacakges)
release-pi:
	lua profile_build_scripts/build_release_pi.lua

# RELEASE without OMP and ATLAS
release-no-omp:
	lua profile_build_scripts/build_release_no_omp.lua

#############################################################################

# DEBUG for MACOSX MACPORTS
debug-macports:
	lua profile_build_scripts/build_debug_macports.lua

# DEBUG for MACOSX HOMEBREW
debug-homebrew:
	lua profile_build_scripts/build_debug_homebrew.lua

# DEBUG with OMP and MKL
debug-mkl:
	lua profile_build_scripts/build_debug_mkl.lua

# DEBUG with OMP and ATLAS
debug-atlas:
	lua profile_build_scripts/build_debug_atlas.lua

# DEBUG without OMP and ATLAS
debug-no-omp:
	lua profile_build_scripts/build_debug_no_omp.lua

# DEBUG with CUDA and MKL
debug-cuda-mkl:
	lua profile_build_scripts/build_debug_cuda_and_mkl.lua

# DEBUG for raspberry-pi (subset of packages)
debug-pi:
	lua profile_build_scripts/build_debug_pi.lua

#############################################################################

clean:
	./clean.sh

install:
	make install-$(UNAME)

uninstall:
	make uninstall-$(UNAME)

install-Darwin:
	mkdir -p $(DARWIN_LUALIB)
	install lib/aprilann.so $(DARWIN_LUALIB)
	install bin/april-ann $(BIN)

install-Linux:
	mkdir -p $(LINUX_LUALIB)
	install lib/aprilann.so $(LINUX_LUALIB)
	install bin/april-ann $(BIN)

uninstall-Darwin:
	rm -f $(DARWIN_LUALIB)/aprilann.so
	rm -f $(BIN)/april-ann

uninstall-Linux:
	rm -f $(LINUX_LUALIB)/aprilann.so
	rm -f $(BIN)/april-ann
