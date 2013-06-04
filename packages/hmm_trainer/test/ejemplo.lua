function print_model(m)
	print("name=", m.name)
	print("transitions={")
	for _,t in pairs(m.transitions) do
		printf("{from=%s, to=%s, prob=%s, emission=%s, output=%s, id=%s}\n",
			t.from, t.to, t.prob, t.emission or "nil", t.output or "", t.id or "nil")
	end
	print"}"
	print("initial=", m.initial)
	print("final=", m.final)
end

--[[
x=HMMTrainer:new(100,100,500)

m1={
	name="modelo 1",
	transitions={
		{from="ini", to="a", prob=1, emission=3},
		{from="a", to="a", prob=0.5, emission=3, id="t1"},
		{from="a", to="b", prob=0.5, emission=5, id="t2"},
		{from="b", to="b", prob=0.3, emission=5},
		{from="b", to="fin", prob=0.7}
	},
	initial="ini",
	final="fin"
}

  
m2={
	name="modelo 2",
	transitions={
		{from="ini", to="a", prob=1, emission=7},
		{from="a", to="a", prob=0.7, emission=7},
		{from="a", to="b", prob=0.3, emission=3},
		{from="b", to="b", prob=0.5, emission=3, id="t1"},
		{from="b", to="fin", prob=0.5, id="t2"}
	},
	initial="ini",
	final="fin",
}
--]]


m3={
	name="modelo 3",
	transitions={
		{from="ini", to="x", prob=1, emission=1},
		{from="x", to="x", prob=0.1, emission=1},
		{from="x", to="y", prob=0.9, emission=2},
		{from="y", to="y", prob=0.4, emission=2, id="t1"},
		{from="y", to="z", prob=0.6, emission=3, id="t2"},
		{from="z", to="z", prob=0.2, emission=3},
		{from="z", to="fin", prob=0.8}
	},
	initial="ini",
	final="fin"
}


m_ex=expand_model(m3,"x","y", m1)
x:model(m_ex)
--]]

---[[
tbl={}
tbl.c=HMMTrainer.utils.generate_hmm3st_desc("c",1,0.5,0.5,0.5)
tbl.a=HMMTrainer.utils.generate_hmm3st_desc("a",4,0.5,0.5,0.5)
tbl.s=HMMTrainer.utils.generate_hmm3st_desc("s",7,0.5,0.5,0.5)
tbl.o=HMMTrainer.utils.generate_hmm3st_desc("o",10,0.5,0.5,0.5)

t=HMMTrainer.trainer()
m1=t:model(HMMTrainer.utils.strtable2tree({"casa", "caso", "saca", "saco"}))


mc=t:model(tbl.c)
ma=t:model(tbl.a)
ms=t:model(tbl.s)
mo=t:model(tbl.o)
	
t:add_to_dict(mc)
t:add_to_dict(ma)
t:add_to_dict(ms)
t:add_to_dict(mo)


printf("\n------------\n")
m1c = m1:generate_C_model()
m1c:print_dot()







