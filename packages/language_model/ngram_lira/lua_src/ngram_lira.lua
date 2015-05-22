local function lira_gz_load_function(filename,
				     dictionary,
				     initial_ngram_word,
				     final_ngram_word,
				     extra)
  -- estadistico
  lm_model = ngram.lira.model{
    stream=gzio.open(filename),
    vocabulary=dictionary:getWordVocabulary(),
    final_word=dictionary:getWordId(final_ngram_word),
    fan_out_threshold=extra.fant_out_threshold or 10,
    ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
  }
  return lm_model
end

local function lira_load_function(filename,
				  dictionary,
				  initial_ngram_word,
				  final_ngram_word,
				  extra)
  -- estadistico
  lm_model = ngram.lira.model{
    filename=filename,
    vocabulary=dictionary:getWordVocabulary(),
    final_word=dictionary:getWordId(final_ngram_word),
    fan_out_threshold=extra.fant_out_threshold or 10,
    ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
  }
  return lm_model
end

local function blira_load_function(filename,
				   dictionary,
				   initial_ngram_word,
				   final_ngram_word,
				   extra)
  -- estadistico binario
  lm_model = ngram.lira.model{
    binary=true,
    filename=filename,
    vocabulary=dictionary:getWordVocabulary(),
    final_word=dictionary:getWordId(final_ngram_word),
    ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
  }
  return lm_model
end

language_models.register_by_extension("gz", lira_gz_load_function)
language_models.register_by_extension("lira", lira_load_function)
language_models.register_by_extension("blira", blira_load_function)
