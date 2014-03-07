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
