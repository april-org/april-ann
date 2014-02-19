 package{ name = "LM_interface",
	  version = "1.0",
	  depends = { "util", "dataset", "symbol_scores" },
	  keywords = { },
	  description = "Interface for generic language models",
	  target{
	    name = "init",
	    mkdir{ dir = "include" },
	    mkdir{ dir = "build" },
	  },
	  target{ name = "clean",
		  delete{ dir = "include" },
		  delete{ dir = "build" },
		},
	  target{
	    name = "provide",
	    depends = "init",
	    copy{ file= "c_src/*.h", dest_dir = "include" },
	    provide_bind{ file = "binding/bind_LM_interface.lua.cc", dest_dir = "include" },
	  },
	  target{
	    name = "build",
	    depends = "provide",
	    use_timestamp = true,
	    -- does not need to compile because implemented methods are defined in a template
	    -- object{ 
	    --   file = "c_src/*.cc",
	    --   include_dirs = "${include_dirs}",
	    --   dest_dir = "build",
	    -- },
	    luac{
	      orig_dir = "lua_src",
	      dest_dir = "build",
	    },
	    build_bind{
	      file = "binding/bind_LM_interface.lua.cc",
	      dest_dir = "build",
	    },
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
