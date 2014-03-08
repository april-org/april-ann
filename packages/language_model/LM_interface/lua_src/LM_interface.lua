language_models = language_models or {}

local lang_models_func_tbl = {}

function language_models.register_by_extension(extension, func)
  lang_models_func_tbl[extension] = func
end

function language_models.load(filename,
			      dictionary,
			      initial_ngram_word,
			      final_ngram_word,
			      extra)
  if not extra then extra = {} end
  local lm_model
  local extension = string.get_extension(filename)
  return lang_models_func_tbl[extension](filename,
					 dictionary,
					 initial_ngram_word,
					 final_ngram_word,
					 extra)
end

---------------------------------------------------------------------------

local query_metatable = language_models.query_result.meta_instance

query_metatable.__len = function(self)
  return self:size()
end

query_metatable.__ipairs = function(self)
  local i=0
  return function()
    if i < self:size() then
      i=i+1
      return i,self:get(i)
    end
  end
end

class_extension(language_models.query_result, "iterate",
		query_metatable.__ipairs)

---------------------------------------------------------------------------

local get_metatable = language_models.get_result.meta_instance

get_metatable.__len = function(self)
  return self:size()
end

get_metatable.__ipairs = function(self)
  local i=0
  return function()
    if i < self:size() then
      i=i+1
      return i,self:get(i)
    end
  end
end

class_extension(language_models.get_result, "iterate",
		get_metatable.__ipairs)
