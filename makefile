ALL: release-mkl

document:
	lua -l formiga build_release.lua document

release-macosx:
	lua -l formiga build_release_macosx.lua

release-mkl:
	lua -l formiga build_mkl_release.lua

release:
	lua -l formiga build_release.lua

release-cuda-mkl:
	lua -l formiga build_cuda_and_mkl_release.lua

debug-macosx:
	lua -l formiga build_debug_macosx.lua

debug-mkl:
	lua -l formiga build_mkl_debug.lua

debug:
	lua -l formiga build_debug.lua

debug-cuda-mkl:
	lua -l formiga build_cuda_and_mkl_debug.lua
