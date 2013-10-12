LUALIB=/usr/lib/lua/5.2
BIN=/usr/bin

ALL: release-mkl

document:
	lua build_release.lua document

test-macosx: debug-macosx
	lua build_debug_macosx.lua test

test: test-mkl

test-mkl: debug-mkl
	lua build_mkl_debug.lua test

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

debug-macosx:
	lua build_debug_macosx.lua

debug-mkl:
	lua build_mkl_debug.lua

debug:
	lua build_debug.lua

debug-cuda-mkl:
	lua build_cuda_and_mkl_debug.lua

clean:
	./clean.sh

install:
	install lib/aprilann.so ${LUALIB}
	install bin/april-ann ${BINLIB}

uninstall:
	rm -f ${LUALIB}/aprilann.so
	rm -f ${BIN}/april-ann
