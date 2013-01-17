local luatype = type

function type(x)
    local real_type = luatype(x)
    if real_type == "userdata" then
        local t = getmetatable(x)
        if t ~= nil and t.id ~= nil then
            return t.id
        end
    end
    return real_type
end

