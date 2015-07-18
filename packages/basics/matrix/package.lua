 package{ name = "matrix",
   version = "1.0",
   depends = { "util", "mathcore", "random", "aprilio" },
   keywords = { "matrix" },
   description = "no description available",
   -- targets como en ant
   target{
     name = "init",
     mkdir{ dir = "build" },
     mkdir{ dir = "include" },
   },
   target{ name = "clean",
     delete{ dir = "build" },
     delete{ dir = "include" },
   },
   target{
     name = "test",
     lua_unit_test{
       file={
         "test/test_ger.lua",
         "test/test_gemm.lua",
         "test/test_gemv.lua",
	 "test/test_matrix_inv_solve.lua",
	 "test/test_matrix_fft.lua",
	 "test/test_matrix_math.lua",
	 "test/test_sparse_matrix.lua",
	 "test/test_convolution.lua",
       },
     },
     -- FIXME: make it compile
     -- c_unit_test{
     -- file = { "test/test_matrix.cc" },
     -- },
   },
   target{
     name = "provide",
     depends = "init",
     copy{ file= "c_src/*.h", dest_dir = "include" },
     exec{ command = [[lua binding/generate_matrix_binding.lua]] },
     provide_bind{ file = "binding/bind_matrix.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_complex_float.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_double.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_int32.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_char.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_bool.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_referenced_vector.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_sparse_matrix.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_sparse_matrix_double.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_sparse_matrix_complex_float.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_options.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matrix_ext.lua.cc", dest_dir = "include" },
     copy{ file= "binding/*.h", dest_dir = "include/binding" },
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     object{ 
       file = "c_src/*.cc",
       include_dirs = "${include_dirs}",
       --flags = "-std=c99", not valid for c++!!!
       dest_dir = "build",
     },
     object{ 
       file = "c_src/*.cu",
       dest_dir = "build",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{
	file = "binding/bind_matrix.lua.cc",
	dest_dir = "build",
     },
     build_bind{
        file = "binding/bind_matrix_complex_float.lua.cc",
        dest_dir = "build",
     },
     build_bind{
        file = "binding/bind_matrix_double.lua.cc",
        dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_matrix_int32.lua.cc",
       dest_dir = "build",
     },
     build_bind{
        file = "binding/bind_matrix_char.lua.cc",
        dest_dir = "build",
     },
     build_bind{
        file = "binding/bind_matrix_bool.lua.cc",
        dest_dir = "build",
     },
     build_bind{
        file = "binding/bind_referenced_vector.lua.cc",
        dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_sparse_matrix.lua.cc",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_sparse_matrix_double.lua.cc",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_sparse_matrix_complex_float.lua.cc",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_matrix_options.lua.cc",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_matrix_ext.lua.cc",
       dest_dir = "build",
     },
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
   },
 }
 
 
