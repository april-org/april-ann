 package{ name = "hmm_trainer",
   version = "1.0",
   depends = { "util", "matrix", "lexClass", "tied_model_manager" },
   keywords = { "hmm_trainer" },
   description = "left to right hidden markov model",
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
     --copy{ file= "c_src/dataset.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_hmm_trainer.lua.cc", dest_dir = "include" }
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     object{ 
       file = "c_src/*.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
       debug="yes",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{ 
       file = "binding/bind_hmm_trainer.lua.cc", 
       dest_dir = "build" ,
       debug="yes",
     }
   },
   target{
     name = "document",
     document_src{
       file= {"c_src/*.h", "c_src/*.c", "c_src/*.cc"},
     },
     document_bind{
       file= {"binding/*.lua.cc"}
     },
   },
 }

