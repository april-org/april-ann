dofile("luapkg/formiga.lua")
local postprocess = dofile("luapkg/postprocess.lua")
formiga.build_dir = "build_debug_macports"

local packages = dofile "profile_build_scripts/package_list.lua"
-- table.insert(packages, "rlcompleter") -- AUTOCOMPLETION => needs READLINE
local metadata = dofile "profile_build_scripts/METADATA.lua"

luapkg{
  program_name = "april-ann.debug",
  description = metadata.description,
  version = metadata.version,
  url = metadata.url,
  verbosity_level = 0,  -- 0 => NONE, 1 => ONLY TARGETS, 2 => ALL
  packages = packages,
  version_flags = metadata.version_flags,
  disclaimer_strings = metadata.disclaimer_strings,
  prefix = metadata.prefix,
  global_flags = {
    debug="yes",
    use_lstrip = "no",
    use_readline="yes",
    optimization = "no",
    add_git_metadata = "yes",
    platform = "unix",
    extra_flags={
      "-mtune=native",
      "-msse",
      "-DUSE_XCODE",
      "-F/System/Library/Frameworks/Accelerate.framework",
      "-pg",
      "-DNO_OMP",
      "-fPIC",
    },
    extra_libs={
      "-L/opt/local/lib", -- macports, change if necessary
      "-lpthread",
      "-lpng",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libLAPACK.dylib",
      "-pg",
      "-rdynamic",
      "-fPIC",
    },
    shared_extra_libs={
     "-flat_namespace",
     "-bundle",
      assert(io.popen("pkg-config --cflags --libs 'lua >= 5.2'"):read("*l"))
    },
  },
  
  main_package = package{
    name = "main_package",
    default_target = "build",
    target{
      name = "init",
      mkdir{ dir = "bin" },
      mkdir{ dir = "build" },
      mkdir{ dir = "include" },
    },
    target{
      name = "provide",
      depends = "init",
      copy{ file = formiga.os.compose_dir(formiga.os.cwd,"lua","include","*.h"), dest_dir = "include" }
    },
    target{ name = "clean_all",
      exec{ command = [[find . -name "*~" -exec rm {} ';']] },
      delete{ dir = "bin" },
      delete{ dir = "build" },
      delete{ dir = "build_doc" },
      delete{ dir = "doxygen_doc" },
    },
    target{ name = "clean",
      delete{ dir = "bin" },
      delete{ dir = "build" },
      delete{ dir = "build_doc" },
    },
    target{ name = "document_clean",
      delete{ dir = "build_doc" },
      delete{ dir = "doxygen_doc" },
    },
    target{
      name = "build",
      depends = "provide",
      compile_luapkg_utils{},
      link_main_program{},
      create_static_library{},
      copy_header_files{},
      dot_graph{
	file_name = "dep_graph.dot"
      },
      use_timestamp = true,
    },
    target{
      name = "test",
      depends = "build",
    },
    target{ name = "document",
      echo{"this is documentation"},
      main_documentation{
	dev_documentation = {
	  main_documentation_file = formiga.os.compose_dir("docs","april_dev.dox"),
	  doxygen_options = {
	    GENERATE_LATEX  = 'NO',
	  },
	},
	user_documentation = {
	  main_documentation_file = formiga.os.compose_dir("docs","april_user_ref.dox"),
	  doxygen_options = {
	    GENERATE_LATEX = 'NO',
	  },
	}
      },
    },
  },
}

postprocess(arg, formiga)
