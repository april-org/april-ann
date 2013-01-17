ALL: release-mkl

configure:
	. configure.sh

document: configure
	lua -l formiga build_release.lua document

release-macosx: configure
	lua -l formiga build_release_macosx.lua

release-mkl: configure
	lua -l formiga build_mkl_release.lua

release: configure
	lua -l formiga build_release.lua

release-cuda-mkl: configure
	lua -l formiga build_cuda_and_mkl_release.lua

debug-macosx: configure
	lua -l formiga build_debug_macosx.lua

debug-mkl: configure
	lua -l formiga build_mkl_debug.lua

debug: configure
	lua -l formiga build_debug.lua

debug-cuda-mkl: configure
	lua -l formiga build_cuda_and_mkl_debug.lua
