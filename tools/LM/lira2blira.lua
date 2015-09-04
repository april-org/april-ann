-- se limita a generar un fichero .blira a partir de un fichero .lira
dictionary_filename = arg[1]
lira_filename       = arg[2]
binary_filename     = arg[3]

-- cargar diccionario
dictionary = lexClass.load(io.open(dictionary_filename))
-- cargar lira
lira_model = language_models.load(lira_filename,
                                  dictionary,
                                  '<s>',
                                  '</s>')

-- todo: check de que lo que lo que hemos cargado es un lira, aunque
-- si no lo es el metodo siguiente petara
lira_model:save_binary{
  filename=binary_filename,
  vocabulary=dictionary:getWordVocabulary(),
}


