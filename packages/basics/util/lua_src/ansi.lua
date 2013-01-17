local fgcolors_enabled={
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

local bgcolors_enabled={
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

local colors_disabled={
    ["black"]="",
    ["bright_black"]="",
    ["red"]="",
    ["bright_red"]="",
    ["green"]="",
    ["bright_green"]="",
    ["yellow"]="",
    ["bright_yellow"]="",
    ["blue"]="",
    ["bright_blue"]="",
    ["magenta"]="",
    ["bright_magenta"]="",
    ["cyan"]="",
    ["bright_cyan"]="",
    ["white"]="",
    ["bright_white"]="",
    ["default"]="",
}

ansi={}

function ansi.enable_colors()
    ansi.fg = fgcolors_enabled
    ansi.bg = bgcolors_enabled
end

function ansi.disable_colors()
    ansi.fg = colors_disabled
    ansi.bg = colors_disabled
end

if util.stdout_is_a_terminal() then
    ansi.enable_colors()
else
    ansi.disable_colors()
end


