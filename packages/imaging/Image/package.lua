 package{ name = "Image",
   version = "1.0",
   depends = { "util", "matrix", "affine_transform", "dataset" },
   keywords = { "image" },
   description = "Image processing module",
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
     copy{ file= "c_src/image.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_image.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_image_RGB.lua.cc", dest_dir = "include" },
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp=true,
     object{ 
       file = "c_src/image.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
       --debug = "yes",
     },
     object{ 
       file = "c_src/utilImageFloat.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
       --debug = "yes",
     },
     object{ 
       file = "c_src/floatrgb.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
       --debug = "yes",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{ file = "binding/bind_image.lua.cc", dest_dir = "build" },
     build_bind{ file = "binding/bind_image_RGB.lua.cc", dest_dir = "build" },
   },
   target{
     name = "document",
     document_src{},
     document_bind{},
   },
 }
 
 
