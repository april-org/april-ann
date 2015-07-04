 package{ name = "data_frame",
   version = "1.0",
   depends = { "util", "mathcore", "matrix", "aprilio" },
   keywords = { "data_frame" },
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
         "test/test.lua",
       },
     },
   },
   target{
     name = "provide",
     depends = "init",
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     luac{
       orig_dir = "lua_src",
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
 
 
