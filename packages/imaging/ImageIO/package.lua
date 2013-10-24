 package{ name = "ImageIO",
   version = "1.0",
   depends = { "Image" },
   keywords = { "image reader, image" },
   description = "loads and saves RGB images",
   -- targets como en ant
   target{
     name = "init",
     mkdir{ dir = "build" },
   },
   target{ name = "clean",
     delete{ dir = "build" },
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
 
 
