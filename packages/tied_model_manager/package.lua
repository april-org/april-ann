 package{ name = "tied_model_manager",
   version = "1.0",
   depends = { },
   keywords = { "tied_model_manager" },
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
     name = "build",
     depends = "init",
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
 
 
