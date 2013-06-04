trainer = HMMTrainer.trainer()

desc_ngrama = trainer:model(HMMTrainer.read_arpa({
                                                  filename="mini.arpa",
                                                  vocabulary=lexClass({"<s>","</s>","wa","wb","wc"}),
                                                  phone_vocabulary={"<s>","</s>","wa", "wb", "wc"},
                                                }))

desc_a = trainer:model(HMMTrainer.utils.str2model_desc("wa", {}, "a"))
desc_b = trainer:model(HMMTrainer.utils.str2model_desc("wb", {}, "b"))
desc_c = trainer:model(HMMTrainer.utils.str2model_desc("wc", {}, "c"))


phone_a = trainer:model(HMMTrainer.utils.generate_lr_hmm_desc("a", {1,2}, {0.5,0.5}, {0.0,0.0}))
phone_b = trainer:model(HMMTrainer.utils.generate_lr_hmm_desc("b", {3,4}, {0.5,0.5}, {0.0,0.0}))
phone_c = trainer:model(HMMTrainer.utils.generate_lr_hmm_desc("c", {5,6}, {0.5,0.5}, {0.0,0.0}))
phone_w = trainer:model({ name = "w",
            transitions={{from="ini", to="fin", prob=1, emission=0}},
            initial="ini",
            final="fin" })

phone_ini = trainer:model({ name = "<s>",
            transitions={{from="ini", to="fin", prob=1, emission=0}},
            initial="ini",
            final="fin" })

phone_fin = trainer:model({ name = "</s>",
            transitions={{from="ini", to="fin", prob=1, emission=0}},
            initial="ini",
            final="fin" })



trainer:add_to_dict(desc_ngrama)
trainer:add_to_dict(desc_a)
trainer:add_to_dict(desc_b)
trainer:add_to_dict(desc_c)
trainer:add_to_dict(phone_a)
trainer:add_to_dict(phone_b)
trainer:add_to_dict(phone_c)
trainer:add_to_dict(phone_w)
trainer:add_to_dict(phone_ini)
trainer:add_to_dict(phone_fin)

m = trainer.models["mini.arpa"]:generate_C_model()

m:print_dot()
