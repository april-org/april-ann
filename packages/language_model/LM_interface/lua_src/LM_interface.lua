language_models = language_models or {}

local lang_models_func_tbl = {}

function language_models.register_by_extension(extension, func)
  lang_models_func_tbl[extension] = func
end

function language_models.load(filename,
			      dictionary,
			      initial_word,
			      final_word,
			      extra)
  if not extra then extra = {} end
  local lm_model
  local extension = string.get_extension(filename)
  return lang_models_func_tbl[extension](filename,
					 dictionary,
					 initial_word,
					 final_word,
					 extra)
end

---------------------------------------------------------------------------
local generic_len    = function(self) return self:size() end
local generic_ipairs = function(self)
  local i=0
  return function()
    if i < self:size() then
      i=i+1
      return i,self:get(i)
    end
  end
end
local function set_iterator_metatable(it_class)
  local metatable = it_class.meta_instance
  metatable.__len = generic_len
  metatable.__ipairs = generic_ipairs
  class_extension(it_class, "iterate",metatable.__ipairs)
end
---------------------------------------------------------------------------

set_iterator_metatable(language_models.__query_result__)
set_iterator_metatable(language_models.__get_result__)
set_iterator_metatable(language_models.__next_keys_result__)
