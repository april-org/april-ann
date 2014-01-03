-- compute_loss(INPUT, TARGET)

EPSILON=1e-03

class_extension(matrix,
		"normalize",
		function(self)
		  for sw in self:sliding_window():iterate() do
		    sw:scal(1/sw:sum())
		  end
		  return self
		end)

function check_loss(i,t,l,f,g)
  if f then
    local e,m = l:compute_loss(i,t)
    assert(math.abs(e - f(i,t)) < EPSILON)
  end
  if g then
    local ep = l:gradient(i,t)
    assert(ep:get_matrix():equals(g(i,t),EPSILON))
  end
end

-- CROSS ENTROPY
check_loss(matrix.col_major(20,1):uniformf(0,1,random(1234)):log(),
	   matrix.col_major(20,1):uniform(0,1,random(525)),
	   ann.loss.cross_entropy(1),
	   function(i,t)
	     local a = i:clone():map(t,
				     function(x,y)
				       return y*x
				     end):sum()
	     local b = i:clone():exp():map(t,
					   function(x,y)
					     return (1-y)*(math.log(1-x))
					   end):sum()
	     return (-a-b)/20
	   end,
	   function(i,t)
	     return i:clone():exp():axpy(-1, t)
	   end)

-- MULTICLASS CROSS ENTROPY
check_loss(matrix.col_major(20,4):uniformf(0,1,random(1234)):normalize():log(),
	   dataset.indexed(dataset.matrix(matrix(20,1):uniform(1,4,random(525))),
			   { dataset.identity(4) }):toMatrix():clone("col_major"),
	   ann.loss.multi_class_cross_entropy(4),
	   function(i,t)
	     return -i:clone():cmul(t):sum()/20
	   end,
	   function(i,t)
	     return i:clone():exp():axpy(-1, t)
	   end)

-- MSE
check_loss(matrix.col_major(20,4):uniformf(0,1,random(1234)),
	   matrix.col_major(20,4):uniform(0,1,random(525)),
	   ann.loss.mse(4),
	   function(i,t)
	     return i:clone():axpy(-1,t):pow(2):sum()*0.5/20
	   end,
	   function(i,t)
	     return i:clone():axpy(-1, t)
	   end)

-- MAE
check_loss(matrix.col_major(20,4):uniformf(0,1,random(1234)),
	   matrix.col_major(20,4):uniform(0,1,random(525)),
	   ann.loss.mae(4),
	   function(i,t)
	     return i:clone():axpy(-1,t):abs():sum()/20/4
	   end)

-- ZERO-ONE
check_loss(matrix.col_major(20,4):uniformf(0,1,random(1234)),
	   matrix.col_major(20,1):uniform(1,4,random(525)),
	   ann.loss.zero_one(4),
	   function(i,t)
	     local idx=1
	     local errors=0
	     for sw in i:sliding_window():iterate() do
	       local _,j = sw:max()
	       if j ~= t:get(idx,1) then errors = errors + 1 end
	       idx = idx + 1
	     end
	     return errors/20
	   end)
