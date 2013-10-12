-- TODO: REVISAR EL APANYO PARA COMPILAR USANDO CUDA!!! ES MUY
-- FEO!!!!!! "buscar nvcc y cuda en el codigo"

-----------------------------------------------------------------------
-- FORMIGA
----------------------------------------------------------------------

-- formiga is a software tool for automating software build
-- processes. It is similar to 'apache ant' but is written in Lua

--  root directory
--  |
--  |- lua -> standard lua sources
--  |
--  |- binding -> all files required to manage packages, make bindings, etc.
--  |
--  |- XXX.lua -> configuration file
--  |
--  |- bin -> executables of lua, luac and whatever the toolkit generate
--  |
--  \_ packages     -> in this directory formiga searches "package.lua" files
--     |- packageA
--     |- packageB
--     |
--     |- A_DIRECTORY
--     |  |
--     |  \_ packageC
--     |
--     \_ packageD
--    
-- Every package should have this structure:
--
--    |- package.lua -> configuration file
--    |- c_src   -> C/C++ source files
--    |- lua_src -> lua source files
--    |- binding -> code to bind C/C++ code into lua
--    |- test
--    \_ doc    
--

-- usage: lua -l formiga package [target]
--
-- luapkg{
--    program_name = "blah",
--    packages = { -- list of packages included in the binary
--      packageA,
--      packageB,
--    },
--    global_flags = {
--      debug="no",
--      use_readline="yes",
--      optimization = "yes", -- global flag
--      platform = "unix",
--      extra_flags = "...",
--    },
--    main_package = package{ ... },
-- }

-- every package.lua contains a call to;
--
--  package{ name = "packageA",
--    depends = { "packageB", "packageC", },
--    keywords = { "one", "two" },
--    description = "this is a package",
--    target{ name = "init",
--      mkdir{ dir = "build" },
--      mkdir{ dir = "include" },
--    },
--    target{ name = "clean",
--      delete{ dir = "build" },
--      delete{ dir = "include" },
--    },
--  }
--

dofile("binding/utilformiga.lua")

local commit_count = io.popen("git rev-list HEAD --count"):read("*l")
if not tonumber(commit_count) then commit_count="UNKNOWN" end

-----------------------------------------------------------------
-- table used to avoid using "other" global variables
formiga = {
  
  ---------------------------------------------------------------
  -- operating system specific details
  os = {
    SEPPATH = ':',
    
    -- TODO: cambiar codigo para utilizar esta variable
    SEPDIR = '/',

    -- required because we cannot use 'chdir' from lua
    basedir = ".",

    -- current working directory
    cwd = ".",

    -- other functions are defined below!!!
  },

  ---------------------------------------------------------------
  -- compiler specific details
  compiler = {
    CUDAcompiler = "nvcc",
    CPPcompiler = os.getenv("CXX") or "g++",
    Ccompiler = os.getenv("CC") or "gcc",
    extra_libs = {"-ldl"},
    shared_extra_libs = {"-shared"},
    extra_flags = { string.format("-DGIT_COMMIT=%d", commit_count), },
    language_by_extension = {
      c = "c", cc = "c++", cxx = "c++", CC = "c++", cpp = "c++",
      cu = "nvcc",
    },
    select_language   = "-x",
    destination       = "-o",
    include_dir       = "-I",
    library_inclusion = "-l",
    object_dir        = "-L",
    compile_object    = "-c",
    debug             = "-g",
    optimization      = "-O3",
    wall              = "-Wall -Wextra -Wno-unused",

    -- global flags for all packages
    global_flags = { },

  },

  ---------------------------------------------------------------
  -- other things related to the package system

  packages_subdir = "packages",
  build_dir = "build",
  documentation_build_dir = "build_doc",
  documentation_dest_dir = "doxygen_doc",
  doc_user_refman_dir = "user_refman",
  doc_developer_dir = "developer",
  
  -- list of funcions to be registered with extern C
  lua_dot_c_register_functions = {},

  -- list of C functions that must be executed in order to register
  -- the bindings of every package
  package_register_functions = {},

  -- table to register packages created with package{...}
  package_table = {},

  -- global properties
  global_properties = {},

  -------------------
  -- package system

  -- (direct and indirect) dependencies of every package
  package_dependencies = {},

  pkgconfig_flags      = {},
  pkgconfig_libs       = {},

  package_link_libraries = {},
  package_library_paths = {},

  main_package_name ="main_package",

  -- current package name
  current_package_name = "",

  -- relates directories and name of every package
  dir_to_package = {},
  package_to_dir = {},

  -- auxiliar, used to create tables dir_to_package and package_to_dir. this
  -- variable is set to the name of the directory before performing a
  -- "dofile" to every package.lua
  -- initialized to formiga.os.cwd in formiga.initialize()
  current_package_dir = "",

  -- dependency list for every package, collected from package depends list
  dependencies = {},

  -- set of packages referenced in luapkg
  set_of_packages = {},

  -- given a keyword, this table stores the set of packages that
  -- reference this keyword
  packages_with_keyword = {}, 

  -- graph with packages
  pkg_graph = graph(),

  -- program_name
  program_name = "default",

  -- color output
  color_output = false,
}


-- auxiliary functions

function table.invert(t)
  local n = {}
  for i,j in pairs(t) do n[j] = i end
  return n
end
--
function table.append(t1,t2)
  for _,j in ipairs(t2) do
    table.insert(t1,j)
  end
end

function printverbose(level, ...)
  if formiga.verbosity_level >= level then
    print(...)
  end
end

function printverbosecolor_aux(print_func, level, fgcolor, bgcolor, ...)
  local fgcolors={
    ["black"]="\27[30m",
    ["bright_black"]="\27[1;30m",
    ["red"]="\27[31m",
    ["bright_red"]="\27[1;31m",
    ["green"]="\27[32m",
    ["bright_green"]="\27[1;32m",
    ["yellow"]="\27[33m",
    ["bright_yellow"]="\27[1;33m",
    ["blue"]="\27[34m",
    ["bright_blue"]="\27[1;34m",
    ["magenta"]="\27[35m",
    ["bright_magenta"]="\27[1;35m",
    ["cyan"]="\27[36m",
    ["bright_cyan"]="\27[1;36m",
    ["white"]="\27[37m",
    ["bright_white"]="\27[1;37m",
    ["default"]="\27[0;39m",
  }
  
  local bgcolors={
    ["black"]="\27[40m",
    ["bright_black"]="\27[1;40m",
    ["red"]="\27[41m",
    ["bright_red"]="\27[1;41m",
    ["green"]="\27[42m",
    ["bright_green"]="\27[1;42m",
    ["yellow"]="\27[43m",
    ["bright_yellow"]="\27[1;43m",
    ["blue"]="\27[44m",
    ["bright_blue"]="\27[1;44m",
    ["magenta"]="\27[45m",
    ["bright_magenta"]="\27[1;45m",
    ["cyan"]="\27[46m",
    ["bright_cyan"]="\27[1;46m",
    ["white"]="\27[47m",
    ["bright_white"]="\27[1;47m",
    ["default"]="\27[0;49m",
  }

  if formiga.verbosity_level >= level then
    -- set colors
    if formiga.color_output then
      if fgcolors[fgcolor] ~= nil then
        io.write(fgcolors[fgcolor])
      end
      if bgcolors[bgcolor] ~= nil then
        io.write(bgcolors[bgcolor])
      end
    end
    print_func(...)
    -- reset colors
    if formiga.color_output then
      io.write(fgcolors["default"])
      io.write(bgcolors["default"])
    end
  end
end

function printverbosecolor(level, fgcolor, bgcolor, ...)
  printverbosecolor_aux(print, level, fgcolor, bgcolor, ...)
end

function writeverbosecolor(level, fgcolor, bgcolor, ...)
  printverbosecolor_aux(io.write, level, fgcolor, bgcolor, ...)
end


function string.tokenize(str,sep)
  local sep = sep or ' \t'
  local list = {}
  for token in string.gmatch(str, '[^'..sep..']+') do
    table.insert(list,token)
  end
  return list
end

-- what is done after printing a warning
function pause_warning()
  io.read("*l")
end


-- obtains the timestamp of a given directory
function formiga.os.get_directory_timestamp(path)
  local f=io.popen("find " .. path .. " -type f -printf '%h/%f %T@\n' 2>/dev/null | " ..
		     "grep  -v '.*/\\..*' | grep -v 'gmon.out' | " ..
                     "cut -d' ' -f2 | sort -n | tail -n 1")
  local timestamp=tonumber(f:read("*l"))
  f:close()
  if timestamp == nil then
    timestamp = 0 
  end
  return timestamp
end

-- obtains the timestamp of a given file
function formiga.os.get_file_timestamp(file)
  local f=io.popen("stat -c %Y "..file ..
		     " 2> /dev/null || echo 0")
  local timestamp=tonumber(f:read("*l"))
  f:close()
  printverbose(1, "timestamp ["..file.."] =",timestamp)
  return timestamp
end

-- decomposes a path in path, file, extension
function formiga.os.path_file_extension(cadena) -- funcion auxiliar
  local _,p,f,e
  _,_, p,f,e = string.find(cadena, "(.*)/(.*)%.(.*)")
  return p,f,e
end

-- todo: assure that no "//" is generated if, by mistake, a "/" appear
-- in some of the arguments
-- WARNING: depends on formiga.os.SEPDIR
function formiga.os.compose_dir(...)
  return table.concat({...},formiga.os.SEPDIR)
end

-- currently uses pwd and popen to obtain the cwd
function formiga.os.getcwd () 
  local f,cwd
  local f = io.popen("pwd")
  local cwd = string.gsub(f:read("*a"),"\n","")
  f:close()
  return cwd
end

-- returns the directory where lua.c is
function formiga.os.get_lua_dot_c_path ()
  local f = io.popen("find lua -name lua.c")
  local total = f:read("*a")
  f:close()
  local p,f,e
  p,f,e = formiga.os.path_file_extension(total)
  return p
end

-- execute a command from formiga.os.basedir
-- WARNING: depends on formiga.os.basedir
function formiga.os.execute(command,continueaftererror)
  io.stdout:flush() -- to get things appear in order
  io.stderr:flush() -- to get things appear in order
  local ok,what,resul
  -- la invocacion precedida por un cd basedir
  ok,what,resul = os.execute("cd "..formiga.os.basedir.."; "..command)
  io.stdout:flush() -- to get things appear in order
  io.stderr:flush() -- to get things appear in order
  if resul ~= 0 and not continueaftererror then
    -- report error and stop everything
    error("Error "..(resul or"nil").." in "..formiga.os.basedir..
	    "\nwhen executing command "..command.."\n")
  end
  return ok
end

-- execute popen from formiga.os.basedir
-- WARNING: depends on formiga.os.basedir
function formiga.os.popen(command)
  return io.popen("cd ".. formiga.os.basedir .."; "..command)
end

-- execute popen from formiga.os.basedir
-- WARNING: depends on formiga.os.basedir
function formiga.os.get_popen_output_line(command)
  local fd    = io.popen("cd ".. formiga.os.basedir .."; "..command)
  local resul = fd:read("*l")
  fd:close()
  return resul
end



-- execute popen from formiga.os.basedir
-- WARNING: depends on formiga.os.basedir
function formiga.os.glob(expr)
  if type(expr) == "string" then expr = { expr } end
  local r = {}
  for _,e in ipairs(expr) do
    local f = io.popen("cd ".. formiga.os.basedir ..
			 "; ls "..e.." 2>/dev/null")
    for i in f:lines() do table.insert(r,i) end
    f:close()
  end
  return r
end

-- used to set up some formiga default values
function formiga.initialize ()
  if formiga.is_initialized == nil then
    formiga.is_initialized = true
    
    formiga.the_configuration_file = arg[0]
    formiga.os.cwd = formiga.os.getcwd()
    formiga.global_properties.lua_include_dir =
      formiga.os.compose_dir(formiga.os.cwd,
                             "lua","include")
    formiga.global_properties.bindtemplates_dir =
      formiga.os.compose_dir(formiga.os.cwd,
                             "binding",
                             "bind_templates")
    formiga.global_properties.documentation_dir =
      formiga.os.compose_dir(formiga.os.cwd,
                             formiga.documentation_build_dir)

    formiga.global_properties.build_dir =
      formiga.os.compose_dir(formiga.os.cwd,
                             formiga.build_dir)
    os.execute("mkdir -p "..formiga.global_properties.build_dir)
    os.execute("mkdir -p "..formiga.os.compose_dir(formiga.global_properties.build_dir,"bin"))

    formiga.lua_dot_c_path = formiga.os.get_lua_dot_c_path()
    formiga.lua_path=formiga.os.compose_dir(formiga.os.cwd,"lua","lua-5.2.2")
  end
end

----------------------------------------------------------------------
--                  cosas especificas de Formiga:
----------------------------------------------------------------------

-- can be used where? into a target? into a task????
-- set a mark property ~= nil
function property (t)
  t.__property__ = 1
  return t
end

-- to execute a package
function formiga.exec_package(package_name,target,global_timestamp)
  if not global_timestamp then global_timestamp = 0 end
  local the_package = formiga.package_table[package_name]
  if (the_package ~= nil) then
    -- miramos si la marca temporal es mas reciente que la global
    --if the_package.timestamp > global_timestamp then
    --  the_package.compile_mark = true
    --end
    -- set formiga.basedir to make things execute from package
    -- directory
    formiga.os.basedir = the_package.basedir

    -- add indirect dependencies
    formiga.current_package_name  = the_package.name
    formiga.package_link_libraries[the_package.name] = the_package.link_libraries
    formiga.package_library_paths[the_package.name] = the_package.library_paths
    local aux_depends         = the_package.depends
    local aux_pkgconfig_flags = ""
    local aux_pkgconfig_libs  = ""
    if the_package.pkgconfig_depends then
      local aux = the_package.pkgconfig_depends
      if type(aux) == "table" then aux = table.concat(aux," ") end
      aux_pkgconfig_flags = formiga.os.get_popen_output_line('pkg-config --cflags '..aux)
      aux_pkgconfig_libs  = formiga.os.get_popen_output_line('pkg-config --libs '..aux)
    end
    -- traverse direct dependencies
    if the_package.depends then
      local rev_dep = table.invert(the_package.depends)
      for i,v in ipairs(the_package.depends) do

        -- a direct dependency must be compiled?
        if formiga.package_table[v].compile_mark then
          the_package.compile_mark = true
        end
        if formiga.package_dependencies[v] then
          -- metemos las indirectas ;)
          for j,w in ipairs(formiga.package_dependencies[v]) do
            if not rev_dep[w] then
              -- sin repetir ;)
              table.insert(aux_depends,w)
              rev_dep[w] = true
            end
          end
        end
      end -- for i,v in ipairs(the_package.depends) do
    else -- if the_package.depends
      aux_depends = {}
    end
    
    formiga.package_dependencies[the_package.name] = aux_depends
    formiga.pkgconfig_flags[the_package.name]      = aux_pkgconfig_flags
    formiga.pkgconfig_libs[the_package.name]       = aux_pkgconfig_libs
    --
    --
    if the_package.compile_mark then
      printverbosecolor(1,"yellow", nil, "[package] "..the_package.name,
			"\tCompile: " .. tostring(the_package.compile_mark))
    else
      printverbosecolor(1,"green", nil, "[package] "..the_package.name,
			"\tCompile: " .. tostring(the_package.compile_mark))
    end
    if target == nil then target = the_package.default_target end
    local thetarget = the_package.target_table[target]
    if thetarget == nil then
      print("WARNING: target "..target.." not found")
    else
      if the_package.compile_mark or not thetarget.use_timestamp then
	thetarget.__target__(thetarget)
      end
      table.insert(formiga.lua_dot_c_register_functions,
		   "register_package_lua_and_binding_"..
		     the_package.name)
    end
  end -- if the_package ~= nil
  return the_package.compile_mark
end

-- function to register a package
-- WARNING uses global variable formiga
function package (t)
  formiga.initialize()
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  if formiga.package_table[t.name] ~= nil then
    error("Error: package "..t.name.." already exists. " ..
	    "package_directory='" .. formiga.current_package_dir .. "'\n")
  end
  formiga.package_table[t.name] = t
  -- set up the timestamp
  t.timestamp = formiga.os.get_directory_timestamp(build_dir)
  t.src_timestamp = formiga.os.get_directory_timestamp(formiga.os.basedir)
  -- and the compile_mark
  if t.src_timestamp > t.timestamp then
    t.compile_mark = true
  else
    t.compile_mark = false
  end

  -- relate the package with the directory
  formiga.dir_to_package[formiga.current_package_dir] = t.name
  formiga.package_to_dir[t.name] = formiga.current_package_dir
  -- store dependencies
  if t.name ~= formiga.main_package_name then    
    formiga.dependencies[t.name] = t.depends
    formiga.pkg_graph:add_node(t.name)
    formiga.pkg_graph:connect(t.name, t.depends)
  end
  -- store keywords
  if t.keywords then
    for _,keyword in pairs(t.keywords) do
      if formiga.packages_with_keyword[keyword] == nil then
	formiga.packages_with_keyword[keyword] = {}
      end
      table.insert(formiga.packages_with_keyword[keyword],t.name)
    end
  end
  -- store basedir
  t.basedir = formiga.os.basedir
  
  -- target_table references the set of targets given the target name
  t.target_table = {}
  t.properties = t.properties or {}
  for i,j in ipairs(t) do
    if j.__target__ ~= nil then
      t.target_table[j.name] = j
      j.package = t
    elseif j.__property__ ~= nil then
      for k,r in pairs(j) do
        if k ~= "__property__" then
          t.properties[k] = r
        end
      end
    end
  end
  return t
end

-- what is executed when a target is executed
function formiga.__target__ (t)

  -- ??????????????
  package_register_functions = {}

  --
  printverbosecolor(1,"bright_black", nil, "[target ] "..t.name)
  -- look for dependencies
  if type(t.depends) == "table" then
    printverbose(2,"-- executing dependencies")
    for _,j in ipairs(t.depends) do
      local othertarget = t.package.target_table[j]
      if othertarget == nil then
        error(string.format("Dependency '%s' for target '%s' not found", j, t.name))
      end
      othertarget.__target__(othertarget)
    end
    printverbose(2,"-- recovering target "..t.name)
  end
  -- execute the tasks
  for _,j in ipairs(t) do
    if type(j) == "table" and j.__task__ ~= nil then
      j.__task__(j)
    end
  end
  --
  if t.name == "build" then
    -- solo para targets tipo build
    generate_package_register_file(t.package,package_register_functions)
  end
end

-- every call to target is done inside a package{}
-- receives a table t
function target(t)
  t.__target__ = formiga.__target__
  t.use_timestamp = t.use_timestamp or false
  if type(t.depends) == "string" then
    t.depends = { t.depends }
  end
  -- array part of table t is suposed to be a list of tasks
  -- these tasks are told to be part of this target
  for _,j in ipairs(t) do
    if j.__task__ ~= nil then
      j.target = t
    end
  end
  -- returns the same table t modified
  return t
end

-- WARNING: depends on global formiga.global_properties
function formiga.expand_properties(thing,tbl)  
  if type(thing) == "string" then
    local number
    repeat
      thing,number = 
	string.gsub(thing,'%${(.-)}',
		    function (x)
		      if not tbl[x] and not formiga.global_properties[x] then
			print("Error, property '"..
				x.."' doesn't exist!!")
			pause_warning()
			return ""
		      end
		      return tbl[x] or formiga.global_properties[x]
		    end)
    until number < 1
  end
  return thing
end

----------------------------------------------------------------------
--                          TASK DEFINITIONS
----------------------------------------------------------------------

----------------------------------------------------------------------
--                               ECHO
----------------------------------------------------------------------

function formiga.__echo__ (t)
  printverbosecolor(1, "bright_white", nil, "[echo   ] "..t[1]) -- a diferencia de otros task, level 1
end

function echo (t)
  if type(t) == "string" then
    t = { t }
  end
  t.__task__ = formiga.__echo__
  return t
end

----------------------------------------------------------------------
--                               MKDIR
----------------------------------------------------------------------

function formiga.__mkdir__ (t)
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  local dir = formiga.os.compose_dir(build_dir, t.dir)
  if dir then
    dir = formiga.expand_properties(dir,t.target.package.properties)
    local command = string.format("mkdir -p %s",dir)
    printverbose(2," [mkdir ] "..command) 
    formiga.os.execute(command)
  end
end

function mkdir (t)
  t.__task__ = formiga.__mkdir__
  return t
end

----------------------------------------------------------------------
--                               DELETE
----------------------------------------------------------------------
-- Attributes:
-- dir -> tells the task to "brutally" delete the entire directory tree

function formiga.__delete__ (t)
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  local dir = formiga.os.compose_dir(build_dir, t.dir)
  if dir then
    dir = formiga.expand_properties(dir,t.target.package.properties)
    local command = string.format("rm -rf %s",dir)
    printverbose(2," [delete] "..command) 
    formiga.os.execute(command)
  end
end

function delete (t)
  t.__task__ = formiga.__delete__
  return t
end

----------------------------------------------------------------------
--                               COPY
----------------------------------------------------------------------

function formiga.__copy__ (t)
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  local dest_dir = formiga.os.compose_dir(build_dir,t.dest_dir)
  local file = t.file
  if dest_dir and file then
    dest_dir = formiga.expand_properties(dest_dir,t.target.package.properties)
    file    = formiga.expand_properties(file,t.target.package.properties)
    local command = string.format("cp -rp %s %s",file,dest_dir)
    printverbose(2," [copy  ] "..command) 
    formiga.os.execute(command)
  end
end

function copy (t)
  t.__task__ = formiga.__copy__
  return t
end

----------------------------------------------------------------------
--                               EXEC
----------------------------------------------------------------------
-- Executes a system command

function formiga.__exec__ (t)
  local command = t.command
  if (command) then
    command = formiga.expand_properties(command,t.target.package.properties)
    printverbose(2," [exec]   "..command)
    formiga.os.execute(command)
  end
end

function exec (t)
  t.__task__ = formiga.__exec__
  return t
end


----------------------------------------------------------------------
--                               LUAEXEC
----------------------------------------------------------------------
-- DEPRECATED????????
-- Executes a lua function without arguments

function formiga.__luaexec__ (t) 
  if type(t[1]) == "function" then t[1]() end
end

function luaexec (t)
  t.__task__ = formiga.__luaexec__
  return t
end

----------------------------------------------------------------------
--                               PROGRAM
----------------------------------------------------------------------

-- donde se usa????????????????????
-- Un programa C de cara a compilar requiere la siguiente informacion:
function formiga .__program__ (t)
  local prop = t.target.package.properties
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  local command
  if (t.file == nil) then
    print("[program]  error: file must be specified")
    return
  end
  local thefiles = formiga.os.glob(formiga.expand_properties(t.file,prop))
  for _,thefile in pairs(thefiles) do
    command = { formiga.compiler.CPPcompiler, formiga.compiler.wall,
		table.concat(formiga.compiler.extra_flags, " "),
		table.concat(formiga.version_flags, " ") }
    local path,file,extension,debug,optimization,otherflags
    path,file,extension = formiga.os.path_file_extension(thefile)
    if formiga.compiler.global_flags.debug == "yes" or
    formiga.expand_properties(t.debug,prop) == "yes" then
      table.insert(command,formiga.compiler.debug)
    end
    if global_flags.optimization == "yes" or
    formiga.expand_properties(t.optimization,prop) == "yes" then
      table.insert(command,formiga.compiler.optimization)
    end
    otherflags = formiga.expand_properties(t.flags,prop) or ""
    if string.len(otherflags) > 0 then 
      table.insert(command,otherflags)
    end
    local language = formiga.expand_properties(t.language,prop) or 
      formiga.language_by_extension[extension] or "c"
    table.insert(command,formiga.compiler.select_language.." "..language)
    table.insert(command,thefile) -- like objects but without "-c "
    local dest_dir = formiga.os.compose_dir(build_dir, formiga.expand_properties(t.dest_dir,prop) or path)
    table.insert(command,formiga.compiler.destination.." "
		   ..dest_dir..SEPDIR..file..".o")
    -- directory inclusion
    -- the directory where the file is
    table.insert(command,formiga.compiler.include_dir..path)
    local directory
    if t.include_dirs then
      for _,directory in pairs(t.include_dirs) do
        for w in string.gmatch(formiga.expand_properties(directory,prop),"[^"..formiga.os.SEPPATH.."]+") do
          table.insert(command,formiga.compiler.include_dir..w)
        end
      end
    end
    -- library inclusion
    if t.libraries then
      for _,libr in pairs(t.libraries) do
        for w in string.gmatch(formiga.expand_properties(directory,prop),"w+") do
          table.insert(command,formiga.compiler.library_inclusion..w)
        end
      end
    end
    -- inclusion de objetos
    table.insert(command,formiga.compiler.object_dir..path) -- el propio directorio donde esta el fichero
    local directory
    if t.object_dirs then
      for _,directory in pairs(t.object_dirs) do
        for w in string.gmatch(formiga.expand_properties(directory,prop),"[^"..formiga.os.SEPPATH.."]+") do
          table.insert(command,formiga.compiler.object_dir..w) 
        end
      end
    end
    command = table.concat(command," ")
    printverbose(2," [program] "..command)
    formiga.os.execute(command)
  end
end

function program (t)
  t.__task__ = formiga.__program__
  if type(t.libraries) == "string" then
    t.libraries = { t.libraries }
  end
  if type(t.include_dirs) == "string" then
    t.include_dirs = { t.include_dirs }
  end
  if type(t.object_dirs) == "string" then
    t.include_dirs = { t.object_dirs }
  end
  return t
end

----------------------------------------------------------------------
--                               OBJECT
----------------------------------------------------------------------
-- Compilar librerias

function formiga .__object__ (t)
  local prop = t.target.package.properties
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  
  if (t.file == nil) then
    print("[object]  error: file must be specified")
    return
  end
  for _,tfile in ipairs(t.file) do
    local thefiles = formiga.os.glob(formiga.expand_properties(tfile,prop))
    for _,thefile in ipairs(thefiles) do
      command = { formiga.compiler.CPPcompiler}
      local path,file,extension,debug,optimization,otherflags
      path,file,extension = formiga.os.path_file_extension(thefile)
      if formiga.compiler.global_flags.debug == "yes" or
      formiga.expand_properties(t.debug,prop) == "yes" then
	table.insert(command,formiga.compiler.debug)
      end
      if formiga.compiler.global_flags.optimization == "yes" or
      formiga.expand_properties(t.optimization,prop) == "yes" then
	table.insert(command,formiga.compiler.optimization)
      end
      otherflags = formiga.expand_properties(t.flags,prop) or ""
      if string.len(otherflags) > 0 then 
        table.insert(command,otherflags)
      end
      language = formiga.expand_properties(t.language,prop) or 
        formiga.compiler.language_by_extension[extension] or "c"
      if language == formiga.compiler.CUDAcompiler then
	if formiga.compiler.global_flags.ignore_cuda then
	  language = formiga.compiler.language_by_extension.cc
	  table.insert(command,formiga.compiler.select_language.." "..language)
          table.insert(command, formiga.compiler.wall)
          table.append(command, formiga.compiler.extra_flags)
	  table.append(command, formiga.version_flags)
	else
	  command[1] = formiga.compiler.CUDAcompiler
	  if formiga.compiler.global_flags.debug == "yes" then
	    command[1] = command[1] .. " -g -G3"
          end
	  local flags = {}
	  for _,flag in ipairs(formiga.compiler.extra_flags) do
	    if not string.match(flag, "arch") and not string.match(flag, "sse") then
  	      if formiga.compiler.global_flags.debug ~= "yes" or not string.match(flag, "%-O[0-9]") then
 	        table.insert(flags, flag)
	      end
	    end
	  end
          table.append(command, flags)
	  table.append(command, formiga.version_flags)
	end
      else
	table.insert(command,formiga.compiler.select_language.." "..language)
        table.insert(command, formiga.compiler.wall)
        table.append(command, formiga.compiler.extra_flags)
	table.append(command, formiga.version_flags)
      end
      table.insert(command,formiga.compiler.compile_object.." "..thefile)
      dest_dir = formiga.os.compose_dir(build_dir, formiga.expand_properties(t.dest_dir,prop) or path)
      os.execute("mkdir -p ".. dest_dir)
      table.insert(command,formiga.compiler.destination.." "..dest_dir..formiga.os.SEPDIR..file..".o")
      -- inclusion de librerias
      if t.libraries then
        for _,libr in pairs(t.libraries) do
          for w in string.gmatch(formiga.expand_properties(directory,prop),"w+") do
            table.insert(command,formiga.compiler.library_inclusion..w) 
          end
        end
      end
      -- directory inclusion
      table.insert(command,formiga.compiler.include_dir..path) -- the directory where the file is
      table.insert(command,formiga.compiler.include_dir..formiga.global_properties.lua_include_dir) -- the directory where lua headers are
      if t.include_dirs then
        for _,directory in pairs(t.include_dirs) do
          for w in string.gmatch(formiga.expand_properties(directory,prop),
				 "[^"..formiga.os.SEPPATH.."]+") do
	    if string.sub(w,1,1) == "/" then 
              table.insert(command,formiga.compiler.include_dir..w)
            else
              table.insert(command,formiga.compiler.include_dir..
			     formiga.os.compose_dir(build_dir, w))
            end
          end
        end
      end
      -- Incluimos las dependencias indirectas...
      for i,directory in ipairs(formiga.package_dependencies[formiga.current_package_name]) do
        for w in string.gmatch(formiga.expand_properties(directory,prop),"[^"..formiga.os.SEPPATH.."]+") do
          local basedir = formiga.package_table[w].basedir
          table.insert(command, formiga.compiler.include_dir..
			 formiga.os.compose_dir(formiga.global_properties.build_dir,
						basedir,
						"include"))
          table.insert(command, formiga.pkgconfig_flags[w])
        end
      end
      -- y anyadimos tb las de pkgconfig_flags
      table.insert(command,
		   formiga.pkgconfig_flags[formiga.current_package_name])
      -- creamos y ejecutamos el comando
      command = table.concat(command," ")
      printverbose(2," [object] "..command)
      local ok = formiga.os.execute(command, true)
      if not ok then
	os.execute("rm -Rf " .. build_dir)
	-- error("ERROR")
	os.exit(1)
      end
    end
  end
end

function object (t)
  t.__task__ = formiga.__object__
  if type(t.file) == "string" then
    t.file = { t.file }
  end
  if type(t.include_dirs) == "string" then
    t.include_dirs = { t.include_dirs }
  end
  if type(t.libraries) == "string" then
    t.libraries = { t.libraries }
  end
  return t
end

----------------------------------------------------------------------
--                               LUAC
----------------------------------------------------------------------
--
-- opciones:
--
-- orig_dir <- directorio origen donde compilar *.lua
--
-- dest_dir <- directorio destino donde dejar el .c y el .o
-- en caso de omision, usa el propio directorio origen
--
-- register_function <- nombre de la funcion creada
-- en caso de omision, le da como nombre el lua_register_$$PROYECTO$$

-- para representar un pedazo de codigo binario como cadena a lo C
function formiga.bin2Cstring(bin)
  local aux = {}
  for i = 1,string.len(bin) do
    table.insert(aux,string.format("\\%03o",string.byte(bin,i)))
  end
  return table.concat(aux)
end


function formiga.__luacode__ (t)
  local prop = t.target.package.properties
  local f,command,orig_dir,dest_dir,reg_function
  local thefile,luafiles
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  orig_dir = formiga.expand_properties(t.orig_dir,prop)
  luafiles = t.file or "*.lua"
  if type(orig_dir) ~= "string" then
    error("Error: luac action needs an orig_dir field of type string\n")
  end
  if type(luafiles) == 'string' then luafiles = { luafiles } end
  local files_list = {}
  for _,luafil in ipairs(luafiles) do
    if type(luafil) ~= 'string' then
      error("Error: luac action field orig_dir is wrong\n")
    end
    table.append(files_list,
		 formiga.os.glob(formiga.os.compose_dir(orig_dir,luafil)))
  end
  lua_data = {}
  for _,lua_file in ipairs(files_list) do
    command = "luac -o /dev/null "..lua_file
    formiga.os.execute(command)
    lua_file_path = formiga.os.compose_dir(formiga.os.basedir,
					   lua_file)
    if formiga.compiler.global_flags.use_lstrip == "yes" then
      command = "lstrip "..lua_file_path
      f_lua = io.popen(command)
    else
      f_lua = io.open(lua_file_path)
    end
    table.insert(lua_data, f_lua:read("*a"))
    f_lua:close()
    printverbose(2," [luac  ] "..lua_file)
    io.stdout:flush()
  end
  dest_dir = t.dest_dir or t.orig_dir
  reg_function = t.reg_function or
    "lua_register_"..t.target.package.name
  io.stdout:flush()
  -- TODO: abrir fichero y guardar en 'el el bytecode como cadena C
  -- SUPERTODO: ver si se puede poner la linea ahora comentada:
  thefile = formiga.os.compose_dir(build_dir,
				   dest_dir,reg_function)
  --    thefile =
  --      formiga.os.append_path_basedir(formiga.os.basedir,
  --				     formiga.os.compose_dir(dest_dir,
  --							    reg_function))
  f = io.open(thefile..".c","w")
  local lua_data_string = table.concat(lua_data, "\n\n")
  f:write([[
	      #include <lua.h>
		#include <lauxlib.h>
		int ]]..reg_function..
	  '(lua_State *L) {\nluaL_loadbuffer(L,\n"'..
	    formiga.bin2Cstring(lua_data_string)..'",'..
	    string.len(lua_data_string)..
	    ',"'..t.target.package.name..'");\nlua_call(L,0,0);\nreturn 0;\n}\n')
  f:close()
  command = table.concat({
			   formiga.compiler.Ccompiler,
			   formiga.compiler.compile_object,
			   thefile..".c",
			   formiga.compiler.destination,
			   thefile..".o",
			   "-I lua/include/",
			   table.concat(formiga.compiler.extra_flags, " "),
			   table.concat(formiga.version_flags, " "),
			 }, " ")
  local ok,what,error_resul = os.execute(command)
  if not ok then
    error("Error en el comando: " .. command)
  end
  os.execute("rm "..thefile..".c")
  
  -- seguramente no hace falta generar este fichero .h
  f = io.open(thefile..".h","w")
  f:write("#include <lua.h>\nint "..reg_function.."(lua_State *L);\n")
  f:close()
  
  -- registramos la funcion
  table.insert(package_register_functions,
	       reg_function)
end

function luac (t)
  t.__task__ = formiga.__luacode__
  return t
end

----------------------------------------------------------------------
--                               PROVIDE_BIND
----------------------------------------------------------------------
-- llama al bindeador con la plantilla .h

function formiga.__provide_bind__ (t)
  local thefile, dest_dir,dest_name,path,file,extension
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  thefile = t.file
  if (thefile) then
    path,file,extension = formiga.os.path_file_extension(thefile)
    dest_dir = formiga.os.compose_dir(build_dir, t.dest_dir or path, "binding")
    os.execute("mkdir -p ".. dest_dir)
    dest_name = string.gsub(file,".lua",".h") -- TODO asumimos fichero es .lua.cc
    command = "lua "..
      formiga.os.compose_dir(formiga.os.cwd,"binding","luabind.lua ")..
      formiga.os.compose_dir(dest_dir,dest_name).." "..
      formiga.os.compose_dir(formiga.global_properties.bindtemplates_dir,
			     "luabind_template.h ")..
      thefile
    printverbose(2," [provide_bind] "..command)
    local f = formiga.os.popen(command)
    for r in f:lines() do
      printverbose(2,r) -- para simular formiga.os.execute(command), se puede quitar
      -- todas las lineas de salida estandar que empiecen por "...register:"
      -- tienen el nombre XXX de una funcion
      -- int XXX(luaState *L);
      -- que se registrara en lua.c
      _,_,fu = string.find(r, "^...register:(.*)$")
      if fu then
        table.insert(package_register_functions,
                     fu)
      end
    end
    f:close()
  end
end

function provide_bind (t)
  t.__task__ = formiga.__provide_bind__
  return t
end

----------------------------------------------------------------------
--                               UNITARY TESTING
----------------------------------------------------------------------

function formiga.__execute_script(t)
  local prop = t.target.package.properties
  local april_binary = formiga.os.compose_dir(formiga.global_properties.build_dir,
					      "bin",
					      formiga.program_name)
  if (t.file == nil) then
    print("[execute_script]  error: file must be specified")
    return
  end
  for _,tfile in ipairs(t.file) do
    local thefiles = formiga.os.glob(formiga.expand_properties(tfile,prop))
    for _,thefile in ipairs(thefiles) do
      command = { april_binary, thefile }
      -- creamos y ejecutamos el comando
      command = table.concat(command," ")
      printverbose(2," [execute_script] "..command)
      local ok,what,error_resul = formiga.os.execute(command, true)
      if not ok then
	error("Unitary test '".. thefile .. "' failed: " .. t.target.package.name)
      end
    end
  end
end

function execute_script(t)
  t.__task__ = formiga.__execute_script
  if type(t.file) == "string" then
    t.file = { t.file }
  end
  return t
end

----------------------------------------------------------------------
--                               BUILD_BIND
----------------------------------------------------------------------
-- llama al bindeador con la plantilla .cc

function formiga.__build_bind__ (t)
  local thefile, dest_dir,dest_name,path,file,extension
  local pack_dir = formiga.package_to_dir[t.target.package.name]
  local build_dir = formiga.os.compose_dir(formiga.global_properties.build_dir, formiga.os.basedir)
  
  
  thefile = t.file
  if (thefile) then
    path,file,extension = formiga.os.path_file_extension(thefile)
    dest_dir = formiga.os.compose_dir(build_dir, t.dest_dir or path, "binding")
    os.execute("mkdir -p ".. dest_dir)

    dest_name = string.gsub(file,".lua",".cc") -- TODO asumimos fichero es .lua.cc
    dest_cfile = formiga.os.compose_dir(dest_dir,dest_name)
    dest_obj = string.gsub(file,".lua",".o") -- TODO asumimos fichero es .lua.cc
    dest_objfile = formiga.os.compose_dir(dest_dir,dest_obj)

    command = "lua "..
      formiga.os.compose_dir(formiga.os.cwd,"binding","luabind.lua ")..
      dest_cfile.." "..
      formiga.os.compose_dir(formiga.global_properties.bindtemplates_dir,
			     "luabind_template.cc ")..
      thefile

    if formiga.verbosity_level < 2 then
      command = command .. " > /dev/null"
    end
    printverbose(2," [build_bind] "..command)
    formiga.os.execute(command)
    -- ya tenemos el .cc, como lo compilamos?
    
    -- creamos y ejecutamos el comando
    command = {
      formiga.compiler.CPPcompiler,
      formiga.compiler.wall,
      "-c",
      dest_cfile,
      "-o", dest_objfile,
      table.concat(formiga.compiler.extra_flags, " "),
      table.concat(formiga.version_flags, " ")}
    local prop = t.target.package.properties
    local id = "${include_dirs}"..formiga.os.SEPPATH..
      "${lua_include_dir}"..formiga.os.SEPPATH..
      --"include"..formiga.os.SEPPATH..
      formiga.os.compose_dir(build_dir,"include")..formiga.os.SEPPATH..
      formiga.os.compose_dir(build_dir,"include", "binding")..formiga.os.SEPPATH..
      formiga.os.compose_dir(formiga.os.cwd,"binding","c_src")
    for w in string.gmatch(formiga.expand_properties(id,prop),
			   "[^"..formiga.os.SEPPATH.."]+") do
      table.insert(command,"-I"..w) 
    end

    -- y anyadimos tb las de pkgconfig_flags
    table.insert(command,
		 formiga.pkgconfig_flags[formiga.current_package_name])

    if formiga.compiler.global_flags.debug == "yes" or
    formiga.expand_properties(t.debug,prop) == "yes" then
      table.insert(command,"-g") 
    end
    if formiga.compiler.global_flags.optimization == "yes" or
    formiga.expand_properties(t.optimization,prop) == "yes" then
      table.insert(command,"-O3")
    end
    otherflags = formiga.expand_properties(t.flags,prop) or ""
    if string.len(otherflags) > 0 then 
      table.insert(command,otherflags)
    end
    -- Incluimos las dependencias indirectas...
    for i,directory in ipairs(formiga.package_dependencies[formiga.current_package_name]) do
      for w in string.gmatch(formiga.expand_properties(directory,prop),
			     "[^"..formiga.os.SEPPATH.."]+") do   
        local basedir = formiga.package_table[w].basedir
        table.insert(command, "-I"..
		       formiga.os.compose_dir(formiga.global_properties.build_dir,
					      basedir,
					      "include"))
        table.insert(command, "-I"..
		       formiga.os.compose_dir(formiga.global_properties.build_dir,
					      basedir,
					      "include", "binding"))
      end
    end
    --
    
    command = table.concat(command," ")
    printverbose(2,"          "..command)
    local ok,what,error_resul = formiga.os.execute(command, true)
    if not ok then
      os.execute("rm -Rf " .. build_dir)
      -- error("ERROR")
      os.exit(1)
    end

  end
end

function build_bind (t)
  t.__task__ = formiga.__build_bind__
  return t
end

----------------------------------------------------------------------
--                           CREATE_STATIC_LIBRARY
----------------------------------------------------------------------

function formiga.__create_static_library__ (t)
  printverbosecolor(1, "bright_blue", nil, "Creating static library...")
  local lib_name = "lib"..formiga.program_name..".a"
  local dest_dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                          "lib")
  local destination = formiga.os.compose_dir(dest_dir, lib_name)

  os.execute("mkdir -p "..dest_dir)
  os.execute("ar rcs "..destination.." "..formiga.get_all_non_binding_objects())
end

function create_static_library (t)
  t.__task__ = formiga.__create_static_library__
  return t
end

----------------------------------------------------------------------
--                           COPY_HEADER_FILES
----------------------------------------------------------------------

function formiga.__copy_header_files__ (t)
  local headers,dir,files
  printverbosecolor(1, "bright_blue", nil, "Copying header files...")
  headers = {}  
  for pkg in pairs(formiga.set_of_packages) do
    dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                 "packages",
                                 formiga.package_to_dir[pkg],
                                 "include",
                                 "*")
    dest_dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                      "include","april",pkg)
    os.execute("mkdir -p "..dest_dir)
    files = formiga.os.glob(dir)
    for _,f in ipairs(files) do
      -- copy only regular files
      local tmp = io.open(f, "r")
      if tmp then
        if tmp:read(0) ~= nil then
          os.execute("cp \""..f.."\" "..dest_dir)
        end
        tmp:close()
      end
    end
  end
end

function copy_header_files (t)
  t.__task__ = formiga.__copy_header_files__
  return t
end

----------------------------------------------------------------------
--                           LINK_MAIN_PROGRAM
----------------------------------------------------------------------

function formiga.__link_main_program__ (t)
  -- crear programa ppal
  --formiga.exec_package("luapkg_main","build")
  local module_name = formiga.program_name:gsub("%-","")
  
  local f = io.open(formiga.os.compose_dir(formiga.global_properties.build_dir, "luapkgMain.cc"),"w")
  --
  f:write('#define lua_c\n')
  f:write('extern "C" {\n')
  f:write("#include <unistd.h>\n")
  f:write("#include <stdio.h>\n")
  f:write("#include <string.h>\n")
  f:write('#include <lua.h>\n')
  f:write('#include <lauxlib.h>\n')
  f:write('#include <lualib.h>\n')
  --f:write('#include <locale.h>\n')
  f:write('}\n')
  f:write('#ifndef GIT_COMMIT\n')
  f:write('#define GIT_COMMIT UNKNOWN\n')
  f:write('#endif\n')
  f:write('#define STRINGFY(X) #X\n')
  f:write('#define TOSTRING(X) STRINGFY(X)\n')
  for i,content in ipairs(formiga.disclaimer_strings) do
    f:write('#define DISCLAIMER' .. i .. ' ' .. content .. '\n')
  end
  f:write('const char *__COMMIT_NUMBER__ = TOSTRING(GIT_COMMIT);\n')
  -- 
  f:write('extern "C" {\n')
  for _,funcname in pairs(formiga.lua_dot_c_register_functions) do
    f:write('int '..funcname..'(lua_State *L);\n')
  end
  --
  f:write('int luaopen_' .. module_name .. '(lua_State *L) { \n')
  for _,funcname in pairs(formiga.lua_dot_c_register_functions) do
    f:write('  '..funcname..'(L);\n')
  end
  f:write('  if (isatty(fileno(stdin)) && isatty(fileno(stdout))) {\n')
  for i,_ in ipairs(formiga.disclaimer_strings) do
    f:write('    luai_writestring(DISCLAIMER' .. i .. ','..
	      'strlen(DISCLAIMER'..i..'));\n')
    f:write('    luai_writeline();\n')
  end
  f:write('  }\n')
  f:write("  lua_newtable(L);\n")
  f:write("  lua_pushstring(L, \"APRIL_LOADED\");\n")
  f:write("  lua_pushboolean(L, true);\n")
  f:write("  lua_rawset(L, -3);\n")
  f:write("  return 1;\n")
  f:write('}\n')
  f:write('}\n')
  -- 
  f:write('#define lua_userinit(L)   luaopen_' .. module_name .. '(L)\n')
  --
  f:write('#include <lua.c>\n')
  f:close()
  --
  
  -- Collect all package libraries and generate compiler options.
  local package_library_paths_str=""
  for package_name, paths in pairs(formiga.package_library_paths) do
    for _, path in ipairs(paths) do
      package_library_paths_str = package_library_paths_str..formiga.compiler.object_dir..path.." "
    end
  end
  local package_link_libraries_str=""
  for package_name, libs in pairs(formiga.package_link_libraries) do
    for _, lib in ipairs(libs) do
      package_link_libraries_str = package_link_libraries_str..formiga.compiler.library_inclusion..lib.." "
    end
  end

  local pkgconfig_libs_list = {}
  for _,j in pairs(formiga.pkgconfig_libs) do
    if j ~= "" then table.insert(pkgconfig_libs_list,j) end
  end
  pkgconfig_libs_list = table.concat(pkgconfig_libs_list," ")

  local command = table.concat({ formiga.compiler.CPPcompiler,
				 formiga.compiler.wall,
				 formiga.compiler.destination,
				 formiga.os.compose_dir(formiga.global_properties.build_dir,
                                                        "bin",
							formiga.program_name),
				 formiga.os.compose_dir(formiga.global_properties.build_dir,
							"luapkgMain.cc"),
				 --formiga.compiler.include_dir,
				 formiga.get_all_objects(),
				 formiga.os.compose_dir(formiga.global_properties.build_dir,"binding","c_src","*.o"),
				 formiga.os.compose_dir(formiga.os.cwd,"lua","lib","*.a"),
                                 package_library_paths_str,
                                 package_link_libraries_str,
				 table.concat(formiga.compiler.extra_libs,
					      " "),
				 table.concat(formiga.compiler.extra_flags,
					      " "),
				 table.concat(formiga.version_flags,
					      " "),
				 pkgconfig_libs_list,
				 ' -lm -I'..formiga.os.compose_dir(formiga.os.cwd,"lua","include")..' -I'..
				   formiga.lua_dot_c_path},
			       " ")
  --
  printverbose(2,'['..command..']')
  io.stdout:flush() -- para que las cosas salgan en un orden apropiado
  io.stderr:flush() -- para que las cosas salgan en un orden apropiado
  formiga.os.execute(command)

  -- We generate a shared library which could be loaded in any Lua interperter
  local shared_lib_dest_dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
						     "lib")
  os.execute("mkdir -p " .. shared_lib_dest_dir)
  local command = table.concat({ formiga.compiler.CPPcompiler,
				 formiga.compiler.wall,
				 formiga.compiler.destination,
				 string.format("%s/%s.so",
					       shared_lib_dest_dir,
					       module_name,
					       ".so"),
				 formiga.os.compose_dir(formiga.global_properties.build_dir,
							"luapkgMain.cc"),
				 --formiga.compiler.include_dir,
				 formiga.get_all_objects(),
				 formiga.os.compose_dir(formiga.global_properties.build_dir,"binding","c_src","*.o"),
                                 package_library_paths_str,
                                 package_link_libraries_str,
				 table.concat(formiga.compiler.extra_libs,
					      " "),
				 table.concat(formiga.compiler.shared_extra_libs,
					      " "),
				 table.concat(formiga.compiler.extra_flags,
					      " "),
				 table.concat(formiga.version_flags,
					      " "),
				 pkgconfig_libs_list,
				 ' -lm -I'..formiga.os.compose_dir(formiga.os.cwd,"lua","include")..' -I'..
				   formiga.lua_dot_c_path},
			       " ")
  --
  printverbose(2,'['..command..']')
  io.stdout:flush() -- para que las cosas salgan en un orden apropiado
  io.stderr:flush() -- para que las cosas salgan en un orden apropiado
  formiga.os.execute(command)

end

function link_main_program (t)
  t.__task__ = formiga.__link_main_program__
  return t
end

----------------------------------------------------------------------
--                               DOT_GRAPH
----------------------------------------------------------------------

function formiga.__dot_graph__ (t)
  local dotfile = io.open(t.file_name, "w")
  dotfile:write("Digraph dep_graph {\n")
  dotfile:write("rankdir=LR;\n")
  for origpkg,_ in formiga.pkg_graph:nodes_iterator() do
    if origpkg ~= formiga.main_package_name and
    formiga.set_of_packages[origpkg] then
      dotfile:write(origpkg .. " [label=" .. origpkg .. "];\n")      
      for _,destpkg in formiga.pkg_graph:next_iterator(origpkg) do
	dotfile:write(origpkg .. " -> " .. destpkg .. ";\n")
      end
    end
  end
  for pkg,_ in pairs(formiga.set_of_packages) do
    if pkg ~= formiga.main_package_name then
      dotfile:write(pkg .. " [label=" .. pkg .. ", shape=octagon];\n")
    end
  end
  dotfile:write("}\n")
  dotfile:close()
end

function dot_graph (t)
  t.__task__ = formiga.__dot_graph__
  t.file_name = t.file_name or "dep_graph.dot"
  return t
end

----------------------------------------------------------------------
--                               DOCUMENT_SRC
----------------------------------------------------------------------

-- copia todos los fuentes especificados en el mismo path de
-- build_doc

function formiga.__document_src__ (t)
  local pack_dir = formiga.package_to_dir[t.target.package.name]
  local group = t.group or pack_dir
  if type(group) == 'string' then
    group = string.tokenize(pack_dir,formiga.os.SEPDIR)
  end
  if type(group) ~= 'table' then
    print("WARNING: document_src group field of wrong type")
    pause_warning()
    return
  end
  local str_begin = ""
  local str_end   = ""
  local lengroup = #group
  if (lengroup > 0) then
    local aux_begin = {}
    local aux_end   = {}
    for _,namegroup in ipairs(group) do
      table.insert(aux_begin,"/// \\addtogroup "..namegroup)
      table.insert(aux_begin,"/// @{")
      table.insert(aux_end,  "/// @}")
    end      
    table.insert(aux_begin,"")
    table.insert(aux_end,  "")
    str_begin = table.concat(aux_begin,"\n")
    str_end   = table.concat(aux_end,"\n")
  end
  t.file = t.file or { "c_src/*.h","c_src/*.cc","c_src/*.c", "doc/*_dev.dox", "doc/*_both.dox" }
  local thefiles = formiga.os.glob(formiga.expand_properties(t.file,prop))
  for _,thefile in pairs(thefiles) do
    --print("thefile",thefile)
    local f = io.open(formiga.os.compose_dir(formiga.os.basedir,thefile),"r")
    local content = f:read("*a")
    f:close()
    local destfile = formiga.os.compose_dir(formiga.documentation_build_dir,
					    formiga.doc_developer_dir,
					    pack_dir,thefile)
    local path = formiga.os.path_file_extension(destfile)
    os.execute("mkdir -p ".. path)
    f = io.open(destfile,"w")
    f:write(str_begin..content..str_end)
    f:close()
  end
end

function document_src (t)
  t.__task__ = formiga.__document_src__
  return t
end

----------------------------------------------------------------------
--                               DOCUMENT_BIND
----------------------------------------------------------------------
-- procesa un fichero .lua.cc para generar cabeceras que
-- enganyen a Doxygen

function formiga.__document_bind__ (t)

  local pack_dir = formiga.package_to_dir[t.target.package.name]
  local group = t.group or pack_dir
  if type(group) == 'string' then
    group = string.tokenize(pack_dir,formiga.os.SEPDIR)
  end
  if type(group) ~= 'table' then
    print("WARNING: document_bind group field of wrong type")
    pause_warning()
    return
  end
  local str_begin = ""
  local str_end   = ""
  local lengroup = #group
  if (lengroup > 0) then
    local aux_begin = {}
    local aux_end   = {}
    for _,namegroup in ipairs(group) do
      table.insert(aux_begin,"/// \\addtogroup "..namegroup)
      table.insert(aux_begin,"/// @{")
      table.insert(aux_end,  "/// @}")
    end
    table.insert(aux_begin,"")
    table.insert(aux_end,  "")
    str_begin = table.concat(aux_begin,"\n")
    str_end   = table.concat(aux_end,"\n")
  end
  t.file = t.file or "binding/*.lua.cc"
  for _,extension in ipairs({".h",".cc"}) do
    local thefiles = formiga.os.glob(formiga.expand_properties(t.file,prop))
    for _,thefile in pairs(thefiles) do
      local newfile  = string.gsub(thefile,".lua.cc",extension)
      local destfile = formiga.os.compose_dir(formiga.os.cwd,
					      formiga.documentation_build_dir,
					      formiga.doc_user_refman_dir,
					      pack_dir,newfile)
      local path = formiga.os.path_file_extension(destfile)
      --print("DESTFILE",destfile)
      os.execute("mkdir -p ".. path)
      command = {
	"lua",
	"-e'lines_information=0'",
	formiga.os.compose_dir(formiga.os.cwd,"binding","luabind.lua "),
	destfile,
	formiga.os.compose_dir(formiga.global_properties.bindtemplates_dir,
			       "luabind_document_template"..extension),
	thefile,
      }
      if formiga.verbosity_level < 2 then
	table.insert(command,">/dev/null")
	table.insert(command,"2>/dev/null")
      end
      command = table.concat(command," ")
      printverbose(0," [document_bind] "..command)
      formiga.os.execute(command)
      local f = io.open(destfile,"r")
      if f == nil then
	error("Error: "..destfile.." file not found\n")
      end
      local content = f:read("*a")
      f:close()
      f = io.open(destfile,"w")
      -- append str_begin and str_end
      f:write(str_begin..content..str_end)
      f:close()
    end
  end
  local aux_file = {"doc/*_user.dox","doc/*_both.dox"}
  local thefiles = formiga.os.glob(formiga.expand_properties(aux_file,prop))
  for _,thefile in pairs(thefiles) do
    --print("thefile",thefile)
    local f = io.open(formiga.os.compose_dir(formiga.os.basedir,thefile),"r")
    local content = f:read("*a")
    f:close()
    local destfile = formiga.os.compose_dir(formiga.documentation_build_dir,
					    formiga.doc_user_refman_dir,
					    pack_dir,thefile)
    local path = formiga.os.path_file_extension(destfile)
    os.execute("mkdir -p ".. path)
    f = io.open(destfile,"w")
    f:write(str_begin..content..str_end)
    f:close()
  end
end

function document_bind (t)
  t.__task__ = formiga.__document_bind__
  return t
end

----------------------------------------------------------------------
--                               DOCUMENT_TEST
----------------------------------------------------------------------
-- hace una copia a lo bestia de carpetas _dox que cuelgan del
-- directorio tests

function formiga.__document_test__ (t)

  local pack_dir = formiga.package_to_dir[t.target.package.name]
  local group = t.group or pack_dir
  if type(group) == 'string' then
    group = string.tokenize(pack_dir,formiga.os.SEPDIR)
  end
  if type(group) ~= 'table' then
    print("WARNING: document_bind group field of wrong type")
    pause_warning()
    return
  end
  local str_begin = ""
  local str_end   = ""
  local lengroup = #group
  if (lengroup > 0) then
    local aux_begin = {}
    local aux_end   = {}
    for _,namegroup in ipairs(group) do
      table.insert(aux_begin,"/// \\addtogroup "..namegroup)
      table.insert(aux_begin,"/// @{")
      table.insert(aux_end,  "/// @}")
    end
    table.insert(aux_begin,"")
    table.insert(aux_end,  "")
    str_begin = table.concat(aux_begin,"\n")
    str_end   = table.concat(aux_end,"\n")
  end
  local aux_file = {"test/*.dox"}
  local thefiles = formiga.os.glob(formiga.expand_properties(aux_file,prop))
  for _,thefile in pairs(thefiles) do
    --print("thefile",thefile)
    local f = io.open(formiga.os.compose_dir(formiga.os.basedir,thefile),"r")
    local content = f:read("*a")
    f:close()
    local destfile = formiga.os.compose_dir(formiga.documentation_build_dir,
					    formiga.doc_user_refman_dir,
					    pack_dir,thefile)
    local path = formiga.os.path_file_extension(destfile)
    os.execute("mkdir -p ".. path)
    f = io.open(destfile,"w")
    --f:write(str_begin..content..str_end)
    f:write(content)
    f:close()
  end
end

function document_test (t)
  t.__task__ = formiga.__document_test__
  return t
end

----------------------------------------------------------------------
--			  DOCUMENT_COPY_FILE
----------------------------------------------------------------------

function formiga.__document_copy_file__ (t)
  local the_type = t.type or formiga.doc_user_refman_dir
  local the_dest = formiga.os.compose_dir(formiga.os.cwd,
					  formiga.documentation_dest_dir,
					  the_type,"html",t.dest_dir)
  command = "mkdir -p ".. the_dest
  printverbose(2," [doc cp] "..command)
  formiga.os.execute(command)
  --
  if t.file then
    local the_files = formiga.os.glob(formiga.expand_properties(t.file,prop))
    for _,thefile in pairs(the_files) do
      command = "cp "..thefile.." "..the_dest
      printverbose(2," [doc cp] "..command)
      formiga.os.execute(command)
    end
  end
end

function document_copy_file (t)
  t.__task__ = formiga.__document_copy_file__
  return t
end

----------------------------------------------------------------------
--                               MAIN_DOCUMENTATION
----------------------------------------------------------------------

-- funcion auxiliar
function formiga.__make_header_doc__ (filename,title,dev_title,user_title,is_dev)
  local if_is_dev = ''
  local if_is_user = ''
  if is_dev == true then
    if_is_dev = ' id="current"'
  else
    if_is_user = ' id="current"'
  end
  local src = 
    '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n'..
    '<html><head><meta http-equiv="Content-Type" content="text/html;charset=utf-8">\n'..
    '<title>'..title..'</title>\n'..
    '<link href="our_doxygen.css" rel="stylesheet" type="text/css">\n'..
    '<link href="tabs.css" rel="stylesheet" type="text/css">\n'..
    '</head><body>\n'..
    '<div class="tabs"><ul>\n'..
    '<li'..if_is_user..'><a href="../../'..formiga.doc_user_refman_dir..
    '/html/index.html"><span>'..user_title..
    '</span></a></li>\n'..
    '<li'..if_is_dev..'><a href="../../'..formiga.doc_developer_dir..
    '/html/index.html"><span>'..dev_title..
    '</span></a></li>\n</ul></div>\n'
  local f= io.open(filename,"w")
  f:write(src)
  f:close()
end

-- funcion auxiliar
function formiga.__make_redirect_page__ (title,wait,urldest)
  return 
    '<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n'..
    '<html><head>\n'..
    '<meta http-equiv="Refresh" content="'..wait..
    '; url='..urldest..'">\n'..
    '</head><title>'..title..'</title>\n'..
    '<body><a href="'..urldest..'">"Pincha aquí"</a>'..
    '</body>\n</html>'
end

function formiga.__main_documentation__ (t)
  -- create redirect page
  local f_dir = formiga.os.compose_dir(formiga.os.cwd,
				       formiga.documentation_dest_dir)
  os.execute("mkdir -p "..f_dir)
  local f = io.open(formiga.os.compose_dir(f_dir,"index.html"),"w")
  -- ojito!!! dest_url es RELATIVO desde f_dir
  local dest_url = formiga.os.compose_dir(formiga.doc_user_refman_dir,
					  "html",
					  "index.html")
  f:write(formiga.__make_redirect_page__(formiga.program_name,0,dest_url))
  f:close()
  --
  for i=1,2 do -- 1 user 2 dev
    local doc_dir_name = ({ formiga.doc_user_refman_dir,formiga.doc_developer_dir})[i]
    local the_table = ({t.user_documentation,t.dev_documentation})[i]
    local doc_file = the_table.main_documentation_file
    local doc_dir = formiga.os.compose_dir(formiga.os.cwd,
					   formiga.documentation_build_dir,
					   doc_dir_name)
    local dest_dir = formiga.os.compose_dir(formiga.os.cwd,
					    formiga.documentation_dest_dir,
					    doc_dir_name)
    if (doc_file) then
      local command = 'mkdir -p '..doc_dir..'; cd '..doc_dir..';ln -s '..
	formiga.os.compose_dir(formiga.os.cwd,doc_file)
      --print("user_documentation: "..command)
      os.execute(command)
    end
    --
    local tbl = the_table.doxygen_options
    if tbl.INPUT == nil then
      tbl.INPUT = doc_dir
    end
    if tbl.PROJECT_NAME == nil then
      tbl.PROJECT_NAME = '"'..formiga.program_name..({ " (user reference manual)",
						       " (developer manual)"})[i]..'"'
    end
    if tbl.OUTPUT_DIRECTORY == nil then
      tbl.OUTPUT_DIRECTORY = dest_dir
    end
    if tbl.STRIP_FROM_PATH == nil then
      tbl.STRIP_FROM_PATH = doc_dir
    end
    if tbl.HTML_HEADER == nil then
      local header_filename = formiga.os.compose_dir(formiga.os.cwd,
						     formiga.documentation_build_dir,
						     doc_dir_name,
						     "header.html")
      --formiga.__make_header_doc__(filename,title,dev_title,user_title,is_dev)
      formiga.__make_header_doc__(header_filename,
				  formiga.program_name,
				  formiga.program_name.." (developer manual)",
				  formiga.program_name.." (user reference manual)",
				  i==2)
      tbl.HTML_HEADER = header_filename
    end
    if tbl.HTML_FOOTER == nil then
      local footer_filename = formiga.os.compose_dir(formiga.os.cwd,
						     formiga.documentation_build_dir,
						     doc_dir_name,
						     "footer.html")
      tbl.HTML_FOOTER = footer_filename
      local f= io.open(footer_filename,"w")
      f:write("\n</BODY>\n</HTML>\n")
      f:close()
    end
    -- copiar el our_doxygen.css
    os.execute(string.format("mkdir -p %s; cp %s %s",
			     formiga.os.compose_dir(dest_dir,"html"),
			     formiga.os.compose_dir(formiga.os.cwd,"binding","our_doxygen.css"),
			     formiga.os.compose_dir(dest_dir,"html")))
    local doxygen_template_filename = 
      formiga.os.compose_dir(formiga.os.cwd,"binding","Doxygen_template")
    local doxygen_aux_file = formiga.os.compose_dir(doc_dir,"doxygen_conf")
    local doxygen_conf = {}
    local f = io.open(doxygen_template_filename,"r")
    if f == nil then
      error("Error: "..doxygen_template_filename.." file not found\n")
    end
    for line in f:lines() do
      -- procesar las lineas y ver si son de la forma
      -- blah = wop
      -- si blah esta en la tabla, reemplazar wop
      local left,right
      _,_,left = string.find(line,"%s*([^%=%s]+)%s*%=.*")
      --print(string.format("line: '%s'",line))
      if left and tbl[left] ~= nil then
	line = left.." = "..tbl[left]
      end
      table.insert(doxygen_conf,line)
    end
    f:close()
    f = io.open(doxygen_aux_file,"w")
    if f == nil then
      error("Error: "..doxygen_template_filename.." file cannot be created\n")
    end
    f:write(table.concat(doxygen_conf,"\n"))
    f:close()
    local command = "doxygen "..doxygen_aux_file
    if formiga.verbosity_level < 2 then
      command = command.." >/dev/null 2>/dev/null"
    end    
    os.execute("mkdir -p "..dest_dir)
    print("ejecutamos",command)
    local ok,what,resul = os.execute(command)
    if not ok then
      -- report error and stop everything
      error("Error "..resul..
	      "\nwhen executing command "..command.."\n")
    end
  end
end


function main_documentation (t)
  t.__task__ = formiga.__main_documentation__
  return t
end

---------------------------------------------------------------------------------

-- ejecuta todos los package.lua encontrados desde la ruta packages
-- WARNING updates global variable formiga.package_table
function formiga.read_all_packages ()
  local f = io.popen('find packages -name package.lua',"r")
  local dirname
  for path in f:lines() do
    _, _, dirname = string.find(path,"packages/(.*)/package.lua")
    printverbose(2,'-- reading package '..dirname)
    -- set these two variables which are used during the dofile
    formiga.current_package_dir = dirname
    formiga.os.basedir = formiga.os.compose_dir("packages",dirname)
    dofile(path)
  end
  f:close()
end

-- updates include_dirs properties of every package
function formiga.add_include_dirs()
  for pkg in pairs(formiga.set_of_packages) do
    local tid = {}
    for _,pkg2 in ipairs(formiga.dependencies[pkg]) do
      table.insert(tid,
		   formiga.os.compose_dir(formiga.global_properties.build_dir,"packages",
					  formiga.package_to_dir[pkg2],
					  "include"))
      table.insert(tid,
		   formiga.os.compose_dir(formiga.global_properties.build_dir,"packages",
					  formiga.package_to_dir[pkg2],
					  "include", "binding"))
    end
    local include_dirs = table.concat(tid,formiga.os.SEPPATH)
    formiga.package_table[pkg].properties.include_dirs = include_dirs
  end
end

-- generates a string with all .o files separated with whitespace
function formiga.get_all_objects()
  local tobj,dir,files
  tobj = {}  
  for pkg in pairs(formiga.set_of_packages) do
    dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                 "packages",
                                 formiga.package_to_dir[pkg],
                                 "build",
                                 "*.o")
    files = formiga.os.glob(dir)
    if #files ~= 0 then table.insert(tobj,dir) end
    dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                 "packages",
                                 formiga.package_to_dir[pkg],
                                 "build", "binding",
                                 "*.o")
    files = formiga.os.glob(dir)
    if #files ~= 0 then table.insert(tobj,dir) end
  end
  return table.concat(tobj," ")
end

-- generates a string with all the .o files which don't come from bindings
function formiga.get_all_non_binding_objects()
  local tobj,dir,files
  tobj = {}  
  for pkg in pairs(formiga.set_of_packages) do
    dir = formiga.os.compose_dir(formiga.global_properties.build_dir,
                                 "packages",
                                 formiga.package_to_dir[pkg],
                                 "build",
                                 "*.o")
    files = formiga.os.glob(dir)
    if #files ~= 0 then table.insert(tobj,dir) end
  end
  return table.concat(tobj," ")
end

-- this function checks if new packages must be added to the list of
-- packages
function formiga.get_all_dependencies ()
  local nodependencies
  repeat
    nodependencies = true
    for pkg in pairs(formiga.set_of_packages) do
      if formiga.dependencies[pkg] == nil then
        print("package "..pkg.." not found")
        return false
      else
        for _,pkg2 in ipairs(formiga.dependencies[pkg]) do
          if formiga.set_of_packages[pkg2] == nil then
            formiga.set_of_packages[pkg2] = 1
            printverbose(2,"adding dependency "..pkg2)
            nodependencies = false
          end
        end
      end
    end
  until nodependencies
  return true
end

------------------------------------------------------------------------

-- funcion para crear el ".o" necesario para registar los fuentes
-- "lua" y bindings de un proyecto
function generate_package_register_file(package,package_register_functions)
  local thefile =
    formiga.os.compose_dir(formiga.global_properties.build_dir,package.basedir,"build",
			   "register_package_lua_and_binding_" .. package.name)  
  local f = io.open(thefile..".c", "w")
  if not f then
    error ("The directory 'build' must be created in package: "..
	     package.name)
  end
  f:write("#include <lua.h>\n#include <lauxlib.h>\n"..
	    "#include <lualib.h>\n\n")
  for i,func in ipairs(package_register_functions) do
    f:write("void " .. func .. "(lua_State *L);\n")
  end
  f:write("\n")
  f:write("void register_package_lua_and_binding_".. package.name ..
	    "(lua_State *L) {\n")
  --f:write("int luaopen_april_".. package.name ..
  --"(lua_State *L) {\n")
  for i,func in ipairs(package_register_functions) do
    f:write("\t"..func .. "(L);\n")
  end
  -- f:write("return 0;\n")
  f:write("}\n")
  f:close()
  
  local command = table.concat({ formiga.compiler.Ccompiler,
				 formiga.compiler.compile_object,
				 thefile..".c",
				 formiga.compiler.destination,
				 thefile..".o",
				 formiga.compiler.include_dir,
				 "lua/include/",
				 table.concat(formiga.compiler.extra_flags,
					      " ")
			       },
			       " ")
  local ok,what,resul = os.execute(command)
  if not ok then
    error("Error en el comando: " .. command)
  end
end


-- treats formiga.compiler.global_flags
function manage_specific_global_flags()
  local t = formiga.compiler.global_flags
  if t.platform == "arm-wince-pocketpc" then
    t.ignore_cuda = true
    formiga.compiler.CPPcompiler = os.getenv("CXX") or "arm-wince-pe-g++"
    formiga.compiler.Ccompiler = os.getenv("CC") or "arm-wince-pe-gcc"
    table.insert(formiga.compiler.extra_flags,
		 "-D__NOTUSE_SIGNAL__")
  elseif t.platform == "unix" then
    t.ignore_cuda = true
    formiga.compiler.CPPcompiler = os.getenv("CXX") or "g++"
    formiga.compiler.Ccompiler = os.getenv("CC") or "gcc"
    table.insert(formiga.compiler.extra_libs,"-ldl")
  elseif t.platform == "unix64+cuda" then
    t.ignore_cuda = false
    table.insert(formiga.compiler.extra_flags, "-DUSE_CUDA")
    table.insert(formiga.compiler.extra_libs,"-lcuda -lcudart -L/usr/local/cuda/lib64")
  end
  if t.use_readline=="yes" then
    table.insert(formiga.compiler.extra_libs,
		 "-lreadline -lhistory -lncurses")
    table.insert(formiga.compiler.extra_flags,
		 "-DLUA_USE_READLINE")
  end
  if t.CPPcompiler then formiga.compiler.CPPcompiler = t.CPPcompiler end
  if t.Ccompiler   then formiga.compiler.Ccompiler = t.Ccompiler end
  if t.extra_libs  then 
    table.append(formiga.compiler.extra_libs,t.extra_libs)
  end
  if t.shared_extra_libs then
    table.append(formiga.compiler.shared_extra_libs,t.shared_extra_libs)
  end
  if t.extra_flags then
    table.append(formiga.compiler.extra_flags,t.extra_flags)
  end
end

------------------------------------------------------------------------
--                              luapkg
------------------------------------------------------------------------

function luapkg (t)
  -- formiga.initialize para obtener formiga.os.basedir, etc.
  formiga.initialize()

  formiga.version_flags    = t.version_flags
  formiga.disclaimer_strings = t.disclaimer_strings

  formiga.verbosity_level = t.verbosity_level or 2
  if os.getenv("TERM")=="xterm" then
    formiga.color_output=true
  end
  
  if arg[1] then
    default_target = arg[1]
  else 
    print("default target: build")
    default_target = "build"
  end
  
  formiga.main_package_name = t.main_package.name

  formiga.read_all_packages()

  -- add all packages in dependency list of main_package
  formiga.dependencies[formiga.main_package_name] = t.packages
  formiga.pkg_graph:add_node(formiga.main_package_name)
  formiga.pkg_graph:connect(formiga.main_package_name,
			    t.packages)
  
  formiga.program_name = t.program_name

  formiga.compiler.global_flags = t.global_flags or {}

  manage_specific_global_flags()

  -- the set of packages
  formiga.set_of_packages = {}
  for _,pkg in pairs(t.packages) do
    formiga.set_of_packages[pkg] = 1
  end
  formiga.set_of_packages[formiga.main_package_name] = 1
  
  if not formiga.get_all_dependencies() then
    print("luapkg dependencies failed")
  else
    
    formiga.add_include_dirs()
    
    local components = 
      formiga.pkg_graph:search_strongly_cc(formiga.set_of_packages)
    
    local rev_top_order = 
      components.components_graph:reverse_top_order(formiga.set_of_packages)
    
    formiga.order_to_process_packages = {}
    for i,j in ipairs(rev_top_order) do
      printverbose(2,"DEBUG Componente ",i,j)
      if components.nodes_graph[j]:size() > 1 then
	print("WARNING: a component of strongly c.c. with "..
		components.nodes_graph[j]:size() .. " nodes:")
	for k,h in components.nodes_graph[j]:nodes_iterator() do
	  print("",k)
	end
	pause_warning()
      end
      for k,h in components.nodes_graph[j]:nodes_iterator() do
	table.insert(formiga.order_to_process_packages,k)
      end
    end
    
    printverbosecolor(1, "blue", nil, "[Reverse Topologic Order] " ..
			table.concat(formiga.order_to_process_packages,","))
    
    -- sacamos el timestamp del script lua y del program_name
    local programnm = formiga.os.compose_dir(formiga.global_properties.build_dir,"bin",formiga.program_name)
    local program_timestamp = formiga.os.get_file_timestamp(programnm)
    local script_timestamp  = 
      formiga.os.get_file_timestamp(formiga.the_configuration_file)
    
    if script_timestamp > program_timestamp then
      -- un 0 en program_timestamp fuerza la compilacion de todo
      program_timestamp = 0
    end
    
    -- ejecutamos :P
    local order = formiga.order_to_process_packages
    local package_string = ""
    local prev_package_string = ""
    for i,pkg in ipairs(order) do
      formiga.exec_package(pkg,default_target, program_timestamp)
      prev_package_string = package_string
      if order[i+1] then package_string = order[i+1] .. " " .. formiga.package_table[order[i+1]].basedir.."/ "
      else package_string = "OK!"
      end
      writeverbosecolor(0, "bright_green", nil, string.format("\r[%3d/%3d (%3.1f%%) done]: %-"
								..tostring(#prev_package_string).."s",
							      i, #order, 100*(i/#order), package_string))
    end
    io.write("\n")
  end
end
