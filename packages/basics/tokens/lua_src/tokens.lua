tokens = tokens or {}

---------------------------------------------------------------------------

class.extend(tokens.null, "to_lua_string",
             function(self)
               return "tokens.null()"
end)

class.extend(tokens.vector.bunch, "to_lua_string",
             function(self, format)
               local str = { "tokens.vector.bunch{" }
               for i,v in self:iterate() do
                 str[#str+1] = util.to_lua_string(v, format)
                 str[#str+1] = ","
               end
               str[#str+1] = "}"
               return table.concat(str)
end)

class.declare_functional_index(tokens.vector.bunch,
                               function(self,key)
                                 local k = tonumber(key)
                                 return (k and self:at(k)) or nil
end)

class.extend_metamethod(tokens.vector.bunch,
                        "__newindex",
                        function(self,key,value)
                          local k = assert(tonumber(key), "Needs a number key")
                          return self:set(k,value)
end)
