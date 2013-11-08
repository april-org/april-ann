local monad_lazy_methods,monad_lazy_class_metatable = class("monad.lazy")

function monad_lazy_class_metatable:__call(value)
  local obj = { value = value }
  return class_instance(obj, self)
end

function monad.lazy.meta_instance:__tostring()
  return "lazy(" .. tostring(self.value) .. ")"
end

function monad_lazy_methods:pass(f)
  if self.value ~= nil then
    return f(self.value)
  else
    return function(...)
      local arg = table.pack(...)
      local aux = arg[1]:pass(f)
      for i=2,#arg do
	aux = aux(arg[i])
      end
      return aux
    end
  end
end

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

function monad.add(left_monad, right_monad)
  local monad_class = get_table_from_dotted_string(get_object_id(left_monad))
  return left_monad:pass(function(left)
			   return right_monad:pass(function(right)
						     return monad_class(left + right)
						   end)
			 end)
end
