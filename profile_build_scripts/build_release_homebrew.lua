dofile("luapkg/formiga.lua")
local postprocess = dofile("profile_build_scripts/postprocess.lua")
formiga.build_dir = "build_release_homebrew"

local packages = dofile "profile_build_scripts/package_list.lua"
-- table.insert(packages, "rlcompleter") -- AUTOCOMPLETION => needs READLINE

luapkg{
  program_name = "april-ann",
  verbosity_level = 0,  -- 0 => NONE, 1 => ONLY TARGETS, 2 => ALL
  packages = packages,
  version_flags = dofile "profile_build_scripts/VERSION.lua",
  disclaimer_strings = dofile "profile_build_scripts/DISCLAIMER.lua",
  global_flags = {
    debug="no",
    use_lstrip = "yes",
    use_readline="yes",
    optimization = "yes",
    platform = "unix",
    extra_flags={
      "-D__HOMEBREW__",
      "-mtune=native",
      "-msse",
      "-DNDEBUG",
      "-DUSE_XCODE",
      "-F/System/Library/Frameworks/Accelerate.framework",
      "-DNO_OMP",
      "-fPIC",
      "-I/usr/local/opt/readline/include", -- homebrew, change if necessary
    },
    extra_libs={
      "-L/usr/local/lib",              -- homebrew, change if necessary
      "-L/usr/local/opt/readline/lib", -- homebrew, change if necessary
      "-lpthread",
      "-lpng",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib",
      "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libLAPACK.dylib",
      "-rdynamic",
      "-fPIC",
    },
    shared_extra_libs={
     "-flat_namespace",
     "-bundle",
     "-llua"
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
      object{ 
	file = formiga.os.compose_dir("binding","c_src","*.cc"),
	include_dirs = "include",
	dest_dir = formiga.global_properties.build_dir,
      },
      link_main_program{},
      create_static_library{},
      create_shared_library{},
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
