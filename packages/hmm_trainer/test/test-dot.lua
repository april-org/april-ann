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

---[[

m1=strtable2tree({"casa", "caso", "saca", "saco","sacar"})

--]]

cadena=model2dot(m1)

print(cadena)
f=io.open("test.dot","w")
f:write(cadena)
f:close()

