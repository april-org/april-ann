 package{ name = "ann.optimizer",
   version = "1.0",
   depends = { "util", "ann", "matrix", "mathcore" },
   keywords = { "ANN optimization algorithms" },
   description = "",
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
     execute_script{
       file={
	 "test/test-digits-sgd.lua",
	 "test/test-digits-rprop.lua",
	 "test/test-digits-cg.lua",
	 "test/test-digits-qprop.lua",
       },
     },
   },
   target{
     name = "provide",
     depends = "init",
     copy{ file= "c_src/*.h", dest_dir = "include" },
     provide_bind{ file = "binding/bind_optimizer.lua.cc", dest_dir = "include" }
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     object{ 
       file = "c_src/*.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_optimizer.lua.cc",
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
 
 
