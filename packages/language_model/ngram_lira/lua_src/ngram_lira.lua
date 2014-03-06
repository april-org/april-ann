ngram = ngram or {}

ngram._lang_models_func_tbl = ngram._lang_models_func_tbl or {}

ngram._lang_models_func_tbl["gz"] =
  function(filename,
	   dictionary,
	   initial_ngram_word,
	   final_ngram_word,
	   extra)
    -- estadistico
    lm_model = ngram.lira{
      command="zcat ".. filename,
      vocabulary=dictionary:getWordVocabulary(),
      fan_out_threshold=10,
      ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
    }
    return lm_model
  end

ngram._lang_models_func_tbl["lira"] =
  function(filename,
	   dictionary,
	   initial_ngram_word,
	   final_ngram_word,
	   extra)
    -- estadistico
    lm_model = ngram.lira{
      filename=filename,
      vocabulary=dictionary:getWordVocabulary(),
      fan_out_threshold=10,
      ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
    }
    return lm_model
  end

ngram._lang_models_func_tbl["blira"] =
  function(filename,
	   dictionary,
	   initial_ngram_word,
	   final_ngram_word,
	   extra)
    -- estadistico binario
    lm_model = ngram.lira.load_binary{
      filename=filename,
      vocabulary=dictionary:getWordVocabulary(),
      ignore_extra_words_in_dictionary = extra.ignore_extra_words_in_dictionary or false
    }
    return lm_model
  end

