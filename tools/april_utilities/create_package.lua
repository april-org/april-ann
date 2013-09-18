local PACKAGE_NAME = arg[1]

local GNU_LICENSE = [[
/*
 * This file is part of APRIL-ANN toolkit (A
 * Pattern Recognizer In Lua with Artificial Neural Networks).
 *
 * Copyright 2013, Salvador España-Boquera, Francisco
 * Zamora-Martinez, Joan Pastor-Pellicer
 *
 * The APRIL-ANN toolkit is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 */
]]



if not PACKAGE_NAME then
    error("A name as argument is required")
end
local root_dir = PACKAGE_NAME

local PACKAGE_INFO = string.format([[
package{ name = "%s",
   version = "1.0",
   depends = { },
   keywords = { },
   description = "A new awesome april package!",
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
     provide_bind{ file = "binding/bind_%s.lua.cc", dest_dir = "include" },
   },
   target{
     name = "build",
     depends = "provide",
     use_timestamp=true,
     object{ 
       file = "c_src/*.cc",
       include_dirs = "${include_dirs}",
       dest_dir = "build",
     },
     luac{
       orig_dir = "lua_src",
       dest_dir = "build",
     },
     build_bind{ file = "binding/bind_%s.lua.cc", dest_dir = "build" },
   },
   target{
     name = "document",
     document_src{},
     document_bind{},
   },
 }
]],PACKAGE_NAME, PACKAGE_NAME, PACKAGE_NAME)
print ("Creating Root dir "..root_dir)
os.execute(string.format("mkdir -p %s %s/binding %s/c_src %s/doc", root_dir, root_dir, root_dir, root_dir))

print("Creating sources")
local fheader = io.open(string.format("%s/c_src/%s.h", root_dir, PACKAGE_NAME), "w")
fheader:write(GNU_LICENSE)
fheader:write(string.format("#ifndef %s_H\n#define %s_H\n\n\n#endif\n",string.upper(PACKAGE_NAME), string.upper(PACKAGE_NAME)))
fheader:close()

local fsource = io.open(string.format("%s/c_src/%s.cc", root_dir, PACKAGE_NAME), "w")
fsource:write(GNU_LICENSE)
fsource:write(string.format("#include \"%s.h\"",PACKAGE_NAME))
fsource:close()

print("Creating Binding...")
local fbinding = io.open(string.format("%s/binding/bind_%s.lua.cc", root_dir, PACKAGE_NAME), "w")
fbinding:write(GNU_LICENSE)
fbinding:write(string.format("//BIND_HEADER_H\n#include <errno.h>\n#include<stdio.h>\n#include \"%s.h\"\n//BIND_END\n",PACKAGE_NAME))
fbinding:close()

print("Setting configuration file...")
local fconf = io.open(string.format("%s/package.lua", PACKAGE_NAME), "w")
fconf:write(PACKAGE_INFO)
fconf:close()
print("Package created")

