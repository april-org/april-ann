ALL: release-mkl

document:
	lua -l formiga build_release.lua document

test-macosx:
	lua -l formiga build_release_macosx.lua test

test-mkl:
	lua -l formiga build_mkl_release.lua test

test:
	lua -l formiga build_release.lua test

test-cuda-mkl:
	lua -l formiga build_cuda_and_mkl_release.lua test

release-macosx:
	lua -l formiga build_release_macosx.lua

release-mkl:
	lua -l formiga build_mkl_release.lua

release:
	lua -l formiga build_release.lua

release-cuda-mkl:
	lua -l formiga build_cuda_and_mkl_release.lua

release-experimental:
	lua -l formiga build_experimental_release.lua

debug-macosx:
	lua -l formiga build_debug_macosx.lua

debug-mkl:
	lua -l formiga build_mkl_debug.lua

debug:
	lua -l formiga build_debug.lua

debug-cuda-mkl:
	lua -l formiga build_cuda_and_mkl_debug.lua

debug-experimental:
	lua -l formiga build_experimental_debug.lua
