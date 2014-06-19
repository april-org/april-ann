package{ name = "ngram.lira",
   version = "1.0",
   depends = { "util", "language_models" },
   keywords = { },
   description = "no description available",
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
	 "test/test_ppl_ngramlira.lua",
       },
     },
   },
   target{
     name = "provide",
     depends = "init",
     copy{ file= "c_src/*.h", dest_dir = "include" },
     provide_bind{ file = "binding/bind_ngram_lira.lua.cc", dest_dir = "include" },
     provide_bind{ file = "binding/bind_arpa2lira.lua.cc", dest_dir = "include" }
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp=true,
     object{ 
       file = {"c_src/*.cc",},
       include_dirs = "${include_dirs}",
       --flags = "-std=c99", not valid for c++!!!
       dest_dir = "build",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_ngram_lira.lua.cc",
       dest_dir = "build",
     },
     build_bind{
       file = "binding/bind_arpa2lira.lua.cc",
       dest_dir = "build",
     }
   },
   target{
     name = "document",
     document_src{
     },
     document_bind{
     },
--      document_copy_file{
--        type="user_refman",
--        file = "doc/mini_sorted.png",
--        dest_dir = "ngram_lira",
--      },
   },
 }
 
 
