require "socket"
-- module master
master = {}

local MAX_NUMBER_OF_ATTEMPS = 20

------------------------------------------------------------
------------------- WORKERS HANDLER ------------------------
------------------------------------------------------------

local worker_methods,worker_class_metatable = class("master.worker")

function worker_class_metatable:__call(name,address,port,nump,mem)
  local obj = { name=name, address=address, port=port, nump=nump, mem=mem,
		is_dead=false, number_of_attemps=0 }
  obj.rel_mem_nump = mem/nump
  obj.load_avg = 0
  return class_instance(obj,self,true)
end

local function check_error(p,msg)
  if not p then fprintf(io.stderr, "ERROR: %s\n", msg) end
  return p
end

function worker_methods:connect()
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  s:settimeout(timeout)
  if not check_error(s,error_msg) then self:set_dead() return false end
  ok,error_msg=s:connect(self.address,self.port)
  if not check_error(ok,error_msg) then self:set_dead() return false end
  return s
end

function worker_methods:ping(select_handler, timeout)
  local ok,error_msg,s,data = true
  s,error_msg = socket.tcp()
  s:settimeout(timeout)
  if not check_error(s,error_msg) then self:set_dead() return false end
  ok,error_msg=s:connect(self.address,self.port)
  if not check_error(ok,error_msg) then
    self.number_of_attemps = self.number_of_attemps + 1
    if self.number_of_attemps > MAX_NUMBER_OF_ATTEMPS then self:set_dead() end
    return false
  end
  self.number_of_attemps = 0
  --
  select_handler:send(s, "PING " .. self.name)
  select_handler:receive(s,
			 function(conn, msg)
			   if msg ~= "PONG" then
			     self:set_dead()
			     select_handler:close(s)
			   end
			 end)
  select_handler:send(s, "LOADAVG " .. self.name)
  select_handler:receive(s,
			 function(conn, msg)
			   self.load_avg = tonumber(msg)
			   if not self.load_avg then
			     self:set_dead()
			     select_handler:close(s)
			   elseif not self:dead() then
			     self.rel_mem_nump = math.round(self:get_nump()/self:get_mem())
			   end
			 end)
  select_handler:send(s, "EXIT")
  select_handler:receive(s)
  select_handler:close(s)
end

function worker_methods:update(address,port,nump,mem)
  self.address = address
  self.port    = port
  self.nump    = nump
  self.mem     = mem
end

function worker_methods:get()
  return self.name,self.address,self.port,self.nump,self.mem,self.load_avg
end

function worker_methods:get_name()
  return self.name
end

function worker_methods:get_mem()
  return self.mem
end

function worker_methods:get_nump()
  return math.max(0, self.nump - self.load_avg)
end

function worker_methods:get_rel_mem_nump()
  return self.rel_mem_nump
end

function worker_methods:set_dead()
  self.is_dead = true
end

function worker_methods:dead()
  return self.is_dead
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local task_methods,
task_class_metatable = class("master.task")

-- state stores the working state of the task, it could be STOPPED, PREPARED,
-- MAP, REDUCE, SEQUENTIAL, LOOP, FINISHED
function task_class_metatable:__call(id, script)
  local obj = { id=id, script=script, plan = {}, state = "STOPPED" }
  local f,t
  if script:match(".*%.lua$") then f = loadfile(script, "t")
  else f = load(script, nil, "t") end
  if not f then
    fprintf(io.stderr, "Impossible to load the given task\n")
    return nil
  end
  local ok,t = pcall(f)
  if not ok then
    fprintf(io.stderr, "Impossible to load the given task: %s\n", t)
    return nil
  end
  -- data is a table with pairs of:
  -- { WHATEVER, size }
  --
  -- In some cases WHATEVER is a filename, in others is a Lua string with a
  -- chunk of data, or a table, matrix, dataset, ...  In case that the
  -- split_function is present, it is possible to split the data to fit better
  -- the number of given cores.
  obj.split_function = t.split_function
  obj.data           = t.data
  if not t.data then
    fprintf(io.stderr, "data field not found\n")
  end
  --
  return class_instance(obj,self,true)
end

function task_methods:state() return self.state end

-- the split_function will receive a data table field (with the pair { WHATEVER,
-- size }), and a slice first,last
function task_methods:prepare_plan(logger, select_handler, workers)
  local memory = iterator(ipairs(workers)):select(2):field(mem):reduce(math.add(),0)
  local cores = iterator(ipairs(workers)):select(2):field(nump):reduce(math.add(),0)
  
  -- it is assumed that the data fits in the memory of the cluster, if the work
  -- is correctly divided.
  local data,split_function = self.data,self.split_function
  local size = iterator(ipairs(data)):select(2):field(2):reduce(math.add(),0)
  local rel_size_memory = size/memory
  local plan = { }
  local plan_size = 0
  -- the data is sorted by memory size, workers are sorted by the memory
  -- available per core, in order to give larger data jobs to the machine which
  -- has more memory available by core.
  table.sort(data, function(a,b) return a[2] > b[2] end)
  table.sort(workers, function(a,b) return a:get_rel_mem_nump() > b:get_rel_mem_nump() end)
  local data_i = 1
  if not split_function then
    -- if it is not possible to split the data, then a greedy algorithm
    -- traverses the sorted machines list and the sorted data list, taking first
    -- the larger data and machines
    for i=1,#workers do
      local w = workers[i]
      local wname = w:get_name()
      plan[wname] = {}
      local worker_mem_nump = w:get_rel_mem_nump()
      local worker_job_size = math.ceil(rel_size_memory * worker_mem_nump)
      for i=1,w:get_nump() do
	local free_size = worker_job_size
	plan[wname][i] = {}
	while free_size > 0 and data_i <= #data do
	  table.insert(plan[wname][i], data[data_i][1])
	  free_size = free_size - data[data_i]
	  data_i = data_i + 1
	end -- while free_size and data_i
	plan_size = plan_size + 1
	if data_i > #data then break end
      end -- for each free processor at current worker
    end -- for each worker
  else
    for i=1,#workers do
      local w = workers[i]
      local wname = w:get_name()
      plan[wname] = {}
      local worker_mem_nump = w:get_rel_mem_nump()
      local worker_job_size = math.ceil(rel_size_memory * worker_mem_nump)
      for i=1,w:get_nump() do
	local free_size = worker_job_size
	plan[wname][i] = {}
	while free_size > 0 and data_i <= #data do
	  table.insert(plan[wname][i], data[data_i][1])
	  free_size = free_size - data[data_i]
	  data_i = data_i + 1
	end -- while free_size and data_i
	plan_size = plan_size + 1
	if data_i > #data then break end
      end -- for each free processor at current worker
    end -- for each worker
  end -- if not split_function
  if data_i ~= #data+1 then
    logger:warning("Impossible to prepare the plan")
    return
  end
  -- plan has how the work was divided between all the available workers
  self.plan = plan
  self.plan_size = plan_size
  self.state = "PREPARED"
  return plan,plan_size
end

function task_methods:execute(logger, select_handler, workers, inv_workers)
  if not self.plan or #self.plan == 0 then
    logger:warningf("Imposible to execute a non existing plan\n")
    return
  end
  --
  local count = 0
  local reduce_size = 0
  local reduce_work = {}
  for wname,jobs in pairs(self.plan) do
    local worker = inv_workers[wname]
    for i=1,#jobs do
      worker:start_job(select_handler, i, jobs[i],
		       function(key,value)
			 if not reduce_work[key] then
			   count = count + 1
			   reduce_work[key] = {}
			 end
			 table.insert(reduce_work[key], value)
		       end,
		       function()
			 count = count + 1
			 if count == self.plan_size then
			   count = 0
			   return true
			 end
			 return false
		       end)
    end
  end
end

function worker_methods:start_job(select_handler, taskid,
				  wname, idx, job,
				  map_result,
				  map_counter,
				  reduce_counter)
  local conn = worker:connect()
  assert(conn, "Unexpected problem while connecting with worker " .. wname)
  --
  select_handler:send(conn,
		      string.format("MAP %s %s %d %s", taskid, wname, idx,
				    table.tostring(job)))
  local reduce_works
  select_handler:receive(conn,
			 function(conn, msg)
			   if msg == "MAPPED" then
			     if map_counter() then
			       select_handler:send(conn,
						   string.format("REDUCE blah"))
			       select_handler:receive(conn,
						      function(conn, msg)
							
						      end)
			     end
			   else
			     local key,value = msg:match("^([^%s]*) (.*)$")
			     map_result(key,value)
			   end
			 end)
end
