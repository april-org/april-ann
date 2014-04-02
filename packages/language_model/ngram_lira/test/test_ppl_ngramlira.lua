local path = arg[0]:get_path()
local vocab = lexClass.load(io.open(path .. "vocab"))
local model = language_models.load(path .. "dihana3gram.lira.gz",
                                   vocab, "<s>", "</s>")
local unk_id = -1

language_models.test_set_ppl{ lm = model,
                              vocab = vocab,
                              testset = path .. "frase",
                              debug_flag = 2,
                              use_bcc = true,
                              use_ecc = true
}
