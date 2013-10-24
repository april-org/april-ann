package{
  name = "rlcompleter",
  version = "1.0",
  depends = { "util" },
  keywords = { "rlcompleter" },
  description = "manages autocompletion",
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
    provide_bind{ file = "binding/bind_rlcompleter.lua.cc", dest_dir = "include" },
  },
  target{
    name = "build",
    depends = "provide",
    use_timestamp = true,
    build_bind{
      file = "binding/bind_rlcompleter.lua.cc",
      dest_dir = "build",
    },
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
