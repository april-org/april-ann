local function postprocess(arg, formiga)
  if arg[2] == nil then
    arg[2] = "."
  end
  assert(arg[2]:gsub("%s","") == arg[2])

  if arg[1] ~= "document" and arg[1] ~= "test" then
    formiga.os.execute("mkdir -p "..formiga.os.compose_dir(arg[2], "bin"))
    formiga.os.execute("mkdir -p "..formiga.os.compose_dir(arg[2], "lib"))
    formiga.os.execute("mkdir -p "..formiga.os.compose_dir(arg[2], "include"))
    formiga.os.execute("cp -f "..formiga.os.compose_dir(formiga.build_dir,"bin",formiga.program_name)
                         .." "..formiga.os.compose_dir(arg[2], "bin", formiga.program_name))
    formiga.os.execute("cp -R "..formiga.os.compose_dir(formiga.build_dir,"lib")
                         .." "..arg[2])
    formiga.os.execute("rm -Rf "..formiga.os.compose_dir(arg[2], "include", formiga.program_name))
    formiga.os.execute("mkdir -p "..formiga.os.compose_dir(arg[2], "include", formiga.program_name))
    formiga.os.execute("cp "..formiga.os.compose_dir(formiga.build_dir,"include",formiga.program_name)
                         .."/*/* "..formiga.os.compose_dir(arg[2], "include", formiga.program_name))
    local dir = formiga.os.compose_dir(arg[2],"include",formiga.program_name)
    local f = io.open(formiga.os.compose_dir(dir,formiga.program_name..".h"),"w")
    f:write(([[
#ifndef %s_H
#define %s_H
extern "C" {
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
int luaopen_aprilann(lua_State *L);
}
]]):format(formiga.module_name:upper(),formiga.module_name:upper()))
    local thefiles = formiga.os.glob(formiga.os.compose_dir(arg[2],"include",formiga.program_name,"*"))
    for _,file in ipairs(thefiles) do
      if not file:find(formiga.program_name..".h", nil, true) then
        local basename = string.sub(file, select(2,file:find(dir, nil, true))+2)
        f:write( ('#include "%s"\n'):format(basename) )
      end
    end
    f:write(("#endif // %s_H\n"):format(formiga.module_name:upper()))
    f:close()
  end
end
return postprocess
