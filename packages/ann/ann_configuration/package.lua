 package{ name = "ann_configuration",
   version = "1.0",
   depends = { "random" },
   keywords = { "ANN", "bunch", "cuda" },
   description = "Define ANNs classes to model configuration of some parameters",
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
   },
   target{
     name = "build",
     depends = "provide",
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
   },
 }
 
 
