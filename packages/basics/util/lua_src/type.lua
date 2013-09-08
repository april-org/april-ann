luatype = type

function type(x)
    local real_type = luatype(x)
    return get_object_id(x) or real_type
end
