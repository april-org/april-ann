-- se limita a generar un fichero .lira
ngram.stat.arpa2lira{
  vocabulary      = lexClass.load(io.open(arg[1])),
  limit_vocab     = (arg[5] and lexClass.load(io.open(arg[5]))) or nil,
  input_filename  = arg[2],
  output_filename = arg[3],
  verbosity       = tonumber(arg[4] or 0),
}
