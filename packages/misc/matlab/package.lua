package{ name = "matlab",
   version = "1.0",
   depends = { "matrix", "util" },
   keywords = { },
   link_libraries = { "z" },
   description = "",
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
     copy{ file= "c_src/*.h", dest_dir = "include" },
     provide_bind{ file = "binding/bind_matlab.lua.cc" , dest_dir = "include" },
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp = true,
     object{ 
       file = "c_src/*.cc",
       dest_dir = "build",
       --       flags = "-std=c99",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{ file = "binding/bind_matlab.lua.cc", dest_dir = "build" },
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
   },
 }
