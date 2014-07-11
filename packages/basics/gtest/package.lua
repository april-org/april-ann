package{ name = "gtest",
   version = "1.0",
   depends = { },
   keywords = { },
   description = "Unit testing",
   -- targets como en ant
   target{
     name = "init",
     mkdir{ dir = "build" },
     mkdir{ dir = "include" },
   },
   target{ 
     name = "clean",
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
     use_timestamp = false,
     object{ 
       file = "c_src/*.cc",
       dest_dir = "build",
       -- flags = "-Wmissing-field-initializers",
     },
   },
   target{
     name = "test",
     depends = "build",
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
   },
 }
