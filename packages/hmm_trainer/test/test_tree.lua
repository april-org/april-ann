trainer = HMMTrainer.trainer()

for i,letra in ipairs{"c","a","s","o"} do
  lettermodel = trainer:model{
    name = letra,
    initial = "ini",
    final = "fin",
    transitions = {
      {from="ini", to="1", emission=0, prob=1},
      {from="1", to="2",   emission=i, id=letra.."12", prob=0.5},
      {from="1", to="1",   emission=i, id=letra.."11", prob=0.5},
      {from="2", to="fin", emission=0, id=letra.."2f", prob=1},
    }
  }
  trainer:add_to_dict(lettermodel)
end

vocabulario = {
  "casa", 
  "caso",
}

arbol = HMMTrainer.utils.strtable2tree(vocabulario)

modelarbol = trainer:model(arbol)

modelarbol_c = modelarbol:generate_C_model()

modelarbol_c:print_dot()



