LINUX_LUALIB = /usr/lib/lua/5.2
DARWIN_LUALIB = /opt/local/lib/lua/5.2
BIN = /usr/bin
UNAME = `uname`

ALL: release-mkl

document:
	lua build_release.lua document

test-macosx: debug-macosx
	lua build_debug_macosx.lua test

test: test-debug

test-mkl: debug-mkl
	lua build_mkl_debug.lua test

test-debug: debug
	lua build_debug.lua test

#test:
#	lua build_debug.lua test

#test-cuda-mkl:
#	lua build_cuda_and_mkl_debug.lua test

release-macosx:
	lua build_release_macosx.lua

release-mkl:
	lua build_mkl_release.lua

release:
	lua build_release.lua

release-cuda-mkl:
	lua build_cuda_and_mkl_release.lua

release-pi:
	lua build_release_pi.lua

release-no-omp:
	lua build_release_no_omp.lua

debug-macosx:
	lua build_debug_macosx.lua

debug-mkl:
	lua build_mkl_debug.lua

debug:
	lua build_debug.lua

debug-cuda-mkl:
	lua build_cuda_and_mkl_debug.lua

debug-pi:
	lua build_debug_pi.lua

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
