learning_rate = 0.1

-----------------
-- DOT PRODUCT --
-----------------

w = matrix.col_major(3, 2, {1, 2,
			    3, 4,
			    5, 6})
c = ann.components.dot_product{ input=2, output=3, weights="w" }
c:set_option("learning_rate", learning_rate)
c:build{ weights = { w = ann.connections{ input=2, output=3, w = w } } }
i = matrix.col_major(2, 2, {5, 6,
			    7, 8})
o = c:forward(tokens.matrix( i:clone() )):get_matrix()

if o ~= (i*w:transpose()) then
  print(o)
  print(i*w:transpose())
  error("Error!!!")
end

j = matrix.col_major(2, 3, {1, 2, 3,
			    4, 5, 6})
o = c:backprop(tokens.matrix( j:clone() )):get_matrix()
if o ~= (j*w)then
  print(o)
  print(j*w)
  error("Error!!!")
end

c:update()
w  = w - learning_rate/math.sqrt(2) * j:transpose() * i
cw = c:copy_weights().w:matrix()
if w ~= cw then
  print(cw)
  print(w)
  error("Error!!!")
end

i = matrix.col_major(1, 2, {10, 15})
o = c:forward(tokens.matrix( i:clone() )):get_matrix()

if o ~= (i*w:transpose()) then
  print(o)
  print(i*w:transpose())
  error("Error!!!")
end

j = matrix.col_major(1, 3, {10, 15, 20})
o = c:backprop(tokens.matrix( j:clone() )):get_matrix()
if o ~= (j*w)then
  print(o)
  print(j*w)
  error("Error!!!")
end

c:update()
w  = w - learning_rate * j:transpose() * i
cw = c:copy_weights().w:matrix()
if w ~= cw then
  print(cw)
  print(w)
  error("Error!!!")
end

----------
-- BIAS --
----------

b = matrix.col_major(4, {1, 2, 3, 4})
c = ann.components.bias{ size=4, weights="b" }
c:set_option("learning_rate", learning_rate)
c:build{ weights = { b = ann.connections{ input=1, output=4,
					  w = b:rewrap(4,1) } } }
i = matrix.col_major(4, {5, 6, 7, 8})
o = c:forward(tokens.matrix( i:clone() )):get_matrix()
if o ~= (b+i) then
  print(o)
  print(b+i)
  error("Error!!!")
end

o = c:backprop(tokens.matrix( i:clone():pow(2) )):get_matrix()
if o ~= i:clone():pow(2) then
  print(o)
  print(i:clone():pow(2))
  error("Error!!!")
end

c:update()
current,old = c:copy_weights().b:matrix()
if current:clone("row_major"):rewrap(4)~=(b-i:clone():pow(2)*learning_rate):rewrap(4) then
  print(current)
  print(b - i:clone():pow(2) * learning_rate)
  error("Error!!!")
end

----------
-- COPY --
----------

c = ann.components.copy{ input=4, times=2 }
c:build()
o = c:forward(tokens.matrix( i:clone() )):convert_to_bunch_vector()
if o:size() ~= 2 then
  print(o:size())
  error("Error!!!")
end
map(function(v)
      if i:rewrap(1,4) ~= v:get_matrix() then
	print(v:get_matrix())
	print(i)
	error("Error!!!")
      end
    end, o.iterate, o)

k = tokens.vector.bunch()
k:push_back( tokens.matrix( i:clone():rewrap(1,4) ) )
k:push_back( tokens.matrix( i:clone():pow(2):rewrap(1,4) ) )
o = c:backprop( k ):get_matrix()
if o ~= (i + i:clone():pow(2)):rewrap(1,4) then
  print(o)
  print(i + i:clone():pow(2))
  error("Error!!!")
end

j = matrix.col_major(2, 4, { 0.1, -0.2, 0.1, -0.3,
			     0.4, -0.2, -0.5, 0.6 })

---------------
-- HARD TANH --
---------------

i = matrix.col_major(2, 4, {-0.1, 0.1, 0.8, -0.8,
			    1, -1, 2, -2})
ref = matrix.col_major(2, 4, {-0.1, 0.1, 0.8, -0.8,
			      1, -1, 1, -1})
c = ann.components.actf.hardtanh()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()

if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

----------
-- TANH --
----------

ref = i:clone():scal(0.5):tanh()
c = ann.components.actf.tanh()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

--------------
-- logistic --
--------------

ref = ( (-i):exp() + i:clone():fill(1) ):pow(-1)
c = ann.components.actf.logistic()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

ref = (o - o:clone():pow(2)):cmul(j:clone()):rewrap(2,4)
o = c:backprop( tokens.matrix( j:clone() ) ):get_matrix()

if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

------------------
-- log_logistic --
------------------

ref = ( (-i):exp() + i:clone():fill(1) ):pow(-1):log()
c = ann.components.actf.log_logistic()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

o = c:backprop( tokens.matrix( j:clone() ) ):get_matrix()
if o ~= j then
  print(o)
  print(j)
  error("Error!!!")
end

--------------
-- softplus --
--------------

ref = i:clone():exp():log1p()
c = ann.components.actf.softplus()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

-------------
-- softmax --
-------------

ref = i:clone():exp()
for w in ref:sliding_window():iterate() do w:scal(1/w:sum()) end

c = ann.components.actf.softmax()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

ref = (o - o:clone():pow(2)):cmul(j:clone()):rewrap(2,4)
o = c:backprop( tokens.matrix( j:clone() ) ):get_matrix()

if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

-----------------
-- log_softmax --
-----------------

ref = i:clone():exp()
map(function(i) r=ref:slice({i,1},{1,4}) r:scal(1/r:sum()) end, range,
    1, ref:dim()[1])
ref:log()
c = ann.components.actf.log_softmax()
c:build{ input=4, output=4 }
o = c:forward( tokens.matrix( i:clone() ) ):get_matrix()
if ref ~= o then
  print(o)
  print(ref)
  error("Error!!!")
end

o = c:backprop( tokens.matrix( j:clone() ) ):get_matrix()

if j ~= o then
  print(o)
  print(j)
  error("Error!!!")
end
