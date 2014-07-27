 package{ name = "stats",
   version = "1.0",
   depends = { "util", "matrix", "random" },
   keywords = { "stats" },
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
     name = "provide",
     depends = "init",
     copy{ file= "c_src/*.h", dest_dir = "include" },
     provide_bind{ file = "binding/bind_stats.lua.cc", dest_dir = "include" }
   },
   target{
     name = "test",
     lua_unit_test{
       file={
	 "test/test_bootstrap.lua",
	 "test/test_comb.lua",
	 "test/test_distributions.lua",
	 "test/test-gs-pca.lua",
	 "test/test-zca-whitening.lua",
       },
     },
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     object{ 
       file = "c_src/*.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
       flags = "-std=c++11",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_stats.lua.cc",
       dest_dir = "build",
     }
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
   },
 }

