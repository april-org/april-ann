
-- print"alineamiento inicial"
-- ----------------------------------------------------------------------
-- a = hmm_trainer.initial_emission_alignment({1,2,3,4},10)
-- ----------------------------------------------------------------------
-- for i,j in dataset.matrix(a):patterns() do
--   printf("%2d -> %s\n",i,string.join(j,","))
-- end
-- os.exit(0)

m = matrix.fromString[[
* 4
ascii
0.5 0.2 0.1 0.3
0.4 0.3 0.9 0.6
0.3 0.1 0.2 0.7
0.2 0.4 0.6 0.1
0.5 0.3 0.2 0.5
0.2 0.1 0.1 0.4
]]

m2 = m:clone()
mseq = matrix(m:dim()[1])

for i,j in dataset.matrix(m):patterns() do
  printf("%d -> %s\n",i,string.join(j,","))
end
print"el clon"
for i,j in dataset.matrix(m2):patterns() do
  printf("%d -> %s\n",i,string.join(j,","))
end
print"------------------------------------"

-------------------------------------------
--hmm_trainer.to_log(m)
-------------------------------------------

-- for i,j in dataset.matrix(m):patterns() do
--   printf("%d -> %s\n",i,string.join(j,","))
-- end

t = HMMTrainer.trainer() -- :new{num_emissions=4, num_states=4, num_transitions=6}

m1 = t:model{
	name="test model",
	transitions={
		{from="1", to="2", prob=1,   emission=1, output="de1a2"},
		{from="2", to="2", prob=0.3, emission=2, output="de2a2"},
		{from="2", to="3", prob=0.3, emission=2, output="de2a3"},
		{from="2", to="4", prob=0.4, emission=3, output="de2a4"},
		{from="4", to="2", prob=0.5, emission=0, output="de4a2"},
		{from="4", to="4", prob=0.5, emission=4, output="de4a4"}
	},
	initial="1",
	final="3"
}


print"begin expectation"
t.trainer:begin_expectation()

m1_C = m1:generate_C_model()

print"probamos fb:"
m1_C:forward_backward{ input_emission=m }

print"viterbi"
mprob, str = m1_C:viterbi{
  input_emission      = m,
  do_expectation      = true,
  output_emission     = m,
  output_emission_seq = mseq,
}
print"fin llamada viterbi"
printf("mprob= %f,str: '%s'\n",mprob,str);

print"end expectation"
t.trainer:end_expectation()

mseqds = dataset.matrix(mseq)
print"las emisiones tras aplicar viterbi:"
for i,j in dataset.matrix(m):patterns() do
  printf("%d -> %s (%d)\n",i,string.join(j,","),mseqds:getPattern(i)[1])
end

kk = t.trainer:get_a_priori_emissions()
print(table.concat(kk,","))
