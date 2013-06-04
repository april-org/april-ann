#!/usr/bin/python
import re
import sys

if sys.argv[2]=="p":
	probs=True
else:
	probs=False

f=file(sys.argv[1], 'r')
lines = f.readlines()

sizespec = re.compile('N=([0-9]+)\s+L=([0-9]+)')
node     = re.compile('I=([0-9]+)\s+W=(\S*)')

if probs:
	arc      = re.compile('J=([0-9]+)\s+S=([0-9]+)\s+E=([0-9]+)\s+l=(\S+)')
else:
	arc      = re.compile('J=([0-9]+)\s+S=([0-9]+)\s+E=([0-9]+)')

# find sizespec
i=0
while True:
	m = sizespec.match(lines[i])
	i+=1
	if m:
		break

num_states = int(m.group(1))
num_transitions = int(m.group(2))

#print num_states, num_transitions

state_words={}
# read states
for l in lines[i:i+num_states]:
	id, word = node.match(l).groups()
	state_words[id] = word
	#print "node:", id, word 

i += num_states

#read transitions
transitions=[]
for l in lines[i:i+num_transitions]:
	if probs:
		id, start, end, lprob = arc.match(l).groups()
		transitions.append((id, start, end, state_words[end], lprob))
	else:
		id, start, end = arc.match(l).groups()
		transitions.append((id, start, end, state_words[end]))

	#print "arc:", id, start, end 

#FIXME: Find initial and final states by looking for source and sink states

#print transitions in april HMM desc format
print "return {"
print '\tname = "%s",'%(sys.argv[1])
print "\ttransitions = {"
for t in transitions:
	if probs:
		if t[3]=='!NULL':
			print '\t\t{from="%s", to="%s", emission=0, lprob=%s},'%(t[1], t[2], t[4])
		else:
			print '\t\t{from="%s", to="%s", emission="%s", lprob=%s},'%(t[1], t[2], t[3], t[4])
	else:
		if t[3]=='!NULL':
			print '\t\t{from="%s", to="%s", emission=0, lprob=0},'%(t[1], t[2])
		else:
			print '\t\t{from="%s", to="%s", emission="%s", lprob=0},'%(t[1], t[2], t[3])
print "\t},"
print '\tinitial="0",'
print '\tfinal="1",'
print"}"



