local monad_lazy_methods,monad_lazy_class_metatable = class("monad.lazy")

function monad_lazy_class_metatable:__call(value)
  local obj = { value = value }
  return class_instance(obj, self)
end

function monad.lazy.meta_instance:__call(...)
  local lt = luatype(self.value)
  if lt == "function" then return self.value(...)
  elseif lt ~= "nil" then return self.value
  end
end

function monad.lazy.meta_instance:__tostring()
  return "lazy(" .. tostring(self.value) .. ")"
end

function monad_lazy_methods:pass(f)
  if self.value ~= nil and luatype(self.value) ~= "function" then
    return f(self.value)
  else
    return monad.lazy(function(arg)
			local aux = f
			if luatype(self.value) == "function" then
			  aux = self.value(arg)
			end
			while #arg > 0 do
			  assert(aux, "Incorrect number of arguments")
			  aux = aux(table.remove(arg,1))
			end
			return aux
		      end)
  end
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

local function binary_monad(f)
  return function(arg)
    assert(type(arg) == "table", "Needs a table with monads as argument")
    assert(#arg >= 2, "Incorrect number of arguments")
    local l_monad = table.remove(arg,1)
    local r_monad = table.remove(arg,1)
    local monad_class = get_table_from_dotted_string(get_object_id(l_monad))
    return l_monad:pass(function(l)
			  return r_monad:pass(function(r)
						return monad_class(f(l,r))
					      end)
			end)
  end
end

local function unary_monad(f)
  return function(arg)
    assert(type(arg) == "table", "Needs a table with monads as argument")
    assert(#arg >= 1, "Incorrect number of arguments")
    local l_monad = table.remove(arg,1)
    local monad_class = get_table_from_dotted_string(get_object_id(l_monad))
    return L_monad:pass(function(left)
			  return monad_class(f(left))
			end)
  end
end

--
monad.add  = binary_monad(math.add())
monad.sub  = binary_monad(math.sub())
monad.mul  = binary_monad(math.mul())
monad.div  = binary_monad(math.div())
monad.eq   = binary_monad(math.eq())
monad.le   = binary_monad(math.le())
monad.lt   = binary_monad(math.lt())
monad.ge   = binary_monad(math.ge())
monad.gt   = binary_monad(math.gt())
monad.land = binary_monad(math.land())
monad.lor  = binary_monad(math.lor())
--
monad.lnot = unary_monad(math.lnot())
