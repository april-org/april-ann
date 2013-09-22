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
			   local msg = table.unpack(string.tokenize(msg or ""))
			   if msg ~= "PONG" then
			     self:set_dead()
			     select_handler:close(s)
			   end
			 end)
  select_handler:send(s, "LOADAVG " .. self.name)
  select_handler:receive(s,
			 function(conn, msg)
			   local msg = table.unpack(string.tokenize(msg or ""))
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

function worker_methods:task(select_handler,logger,taskid,script)
  local s = worker_methods:connect()
  if not s then return false end
  select_handler:send( s, string.format("TASK %s %s",taskid,script) )
  select_handler:receive(s,
			 function(conn,msg)
			   local msg = table.concat(string.tokenize(msg))
			   if msg ~= "OK" then return "ERROR" end
			   return "OK"
			 end)
  select_handler:close(s)
end

function worker_methods:do_map(task,select_handler,logger,map_key,job)
  local s = worker_methods:connect()
  if not s then return false end
  select_handler:send( s, string.format("MAP %s %s", map_key,
					table.tostring(job)) )
  -- MAP_RESULT
  select_handler:receive(s,
			 function(conn,msg)
			   local taskid,map_key,result = common.load(logger,msg)
			   if not result then return "ERROR" end
			   local ok = task:process_map_result(taskid,map_key,result)
			   if not ok then return "ERROR" end
			   return "OK"
			 end)
  select_handler:close(s)
end

function worker_methods:do_reduce(task,select_handler,logger,key,value)
  local s = worker_methods:connect()
  if not s then return false end
  select_handler:send(s, string.format("REDUCE %s %s", key, value))
  -- REDUCE_RESULT
  select_handler:receive(s,
			 function(conn,msg)
			   local taskid,key,result = common.load(logger,msg)
			   if not result then return "ERROR" end
			   local ok = task:process_reduce_result(taskid,key,result)
			   if not ok then return "ERROR" end
			   return "OK"
			 end)
  select_handler:close(s)
end

-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------

local task_methods,
task_class_metatable = class("master.task")

-- state stores the working state of the task, it could be STOPPED, PREPARED,
-- ERROR, MAP, MAP_FINISHED, REDUCE, REDUCE_FINISHED, SEQUENTIAL,
-- SEQUENTIAL_FINISHED, LOOP, FINISHED
function task_class_metatable:__call(logger, select_handler, conn, id, script)
  local obj = {
    logger = logger,
    select_handler = select_handler,
    conn=conn,
    id=id,
    script=script,
    map_plan = {},
    map_done = {},
    map_plan_size = 0,
    map_plan_count = 0,
    state = "STOPPED",
    reduction = {},
    reduction_pending_list = {},
    reduce_done = {},
    reduce_size = 0,
    reduce_count = 0,
    reduce_worker = {},
    map_reduce_result = {},
  }
  local t   = common.load(logger, script)
  -- data is a table with pairs of:
  -- { WHATEVER, size }
  --
  -- In some cases WHATEVER is a filename, in others is a Lua string with a
  -- chunk of data, or a table, matrix, dataset, ...  In case that the
  -- split_function is present, it is possible to split the data to fit better
  -- the number of given cores.
  obj.split_function  = t.split_function or function(value,first,last) return value,last-first+1 end
  obj.decode_function = t.decode_function or function(value) return value end
  obj.data            = t.data
  if not t.data then
    fprintf(io.stderr, "data field not found\n")
  end
  --
  return class_instance(obj,self,true)
end

function task_methods:state() return self.state end

-- the split_function will receive a data table field (with the pair { WHATEVER,
-- size }), and a slice first,last

-- This method prepares a static plan for map jobs, so the map is repeated
-- equals during the global loop. If an error occurs, and a worker dies, then
-- the plan will be recomputed
function task_methods:prepare_map_plan(workers)
  local logger,select_handler = self.logger,self.select_handler
  local memory = iterator(ipairs(workers)):select(2):field(mem):reduce(math.add(),0)
  local cores = iterator(ipairs(workers)):select(2):field(nump):reduce(math.add(),0)
  
  -- it is assumed that the data fits in the memory of the cluster, if the work
  -- is correctly divided.
  local data,split_function = self.data,self.split_function
  local size = iterator(ipairs(data)):select(2):field(2):reduce(math.add(),0)
  local rel_size_memory = size/memory
  local map_plan = { }
  local map_plan_size = 0
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
      local worker_mem_nump = w:get_rel_mem_nump()
      local worker_job_size = math.ceil(rel_size_memory * worker_mem_nump)
      for i=1,w:get_nump() do
	local map_key   = wname .. "-" .. i
	local free_size = worker_job_size
	map_plan[map_key] = { worker = w, job = {} }
	local job = map_plan[map_key].job
	while free_size > 0 and data_i <= #data do
	  table.insert(job, data[data_i][1])
	  free_size = free_size - data[data_i]
	  logger:debugf("MAP_JOB %s %d\n", map_key, data_i)
	  data_i = data_i + 1
	end -- while free_size and data_i
	map_plan_size = map_plan_size + 1
	if data_i > #data then break end
      end -- for each free processor at current worker
    end -- for each worker
  else -- if not split_function ... then
    -- the work could be splitted
    error("NOT IMPLEMENTED YET")
  end -- if not split_function ... else
  if data_i ~= #data+1 then
    logger:warning("Impossible to prepare the map_plan")
    self.state = "ERROR"
    return
  end
  -- map_plan has how the work was divided between all the available workers
  self.map_plan = map_plan
  self.map_plan_size = map_plan_size
  self.state = "PREPARED"
  return map_plan,map_plan_size
end

-- ask the workers to do a map job
function task_methods:do_map()
  local logger,select_handler = self.logger,self.select_handler
  if not self.map_plan or #self.map_plan == 0 then
    logger:warningf("Imposible to execute a non existing map_plan\n")
    self.state = "ERROR"
    return
  end
  --
  self.map_plan_count = 0
  for map_key,data in pairs(self.map_plan) do
    local worker = data.worker
    local job    = data.job
    worker:do_map(self, select_handler, logger, map_key, job)
  end
  self.state = "MAP"
  self.map_done = {}
  self.reduction = {}
end

function task_methods:process_map_result(taskid,map_key,result)
  local logger,select_handler = self.logger,self.select_handler
  if taskid ~= self.id then
    logger:warningf("Incorrect task id, found %s, expected %s\n",
		    taskid, self.id)
    return
  end
  if not self.map_plan[map_key] then
    logger:warningf("Incorrect map_key %s\n", map_key)
    return
  end
  if self.map_done[map_key] then
    logger:warningf("Job %s already done\n", map_key)
    return
  end
  -- mark the job as done
  self.map_done[map_key] = true
  self.map_plan_count    = self.map_plan_count + 1
  --
  if not type(result) == "table" then
    logger:warningf("Incorrect map result type, expected a table\n")
    return
  end
  self.reduction = iterator(ipairs(result)):select(2):
  reduction(function(acc,t)
	      acc[t[1]] = table.insert(acc[t[1]] or {}, t[2])
	      return acc
	    end,
	    self.reduction)
  if self.map_plan_count == self.map_plan_size then
    self.state = "MAP_FINISHED"
  end
  return true
end

-- ask the workers to do a map job
function task_methods:do_reduce(workers)
  local logger,select_handler = self.logger,self.select_handler
  if #self.reduction == 0 then
    logger:warningf("Impossible to execute a void reduction\n")
    self.state = "ERROR"
    return
  end
  --
  self.reduction_pending_list = {}
  self.reduce_done = {}
  self.reduce_worker = {}
  local id = 0
  for key,value in pairs(self.reduction) do
    id = id + 1
    local worker = workers[id]
    self.reduce_worker[key] = worker
    if id > #workers then
      table.insert(self.reduction_pending_list, key)
    else
      worker:do_reduce(self, select_handler, logger, key, value)
    end
  end
  self.reduce_size  = id
  self.reduce_count = 0
  self.map_reduce_result = {}
  self.state = "REDUCE"
end

function task_methods:process_reduce_result(taskid,key,result)
  local logger,select_handler = self.logger,self.select_handler
  if taskid ~= self.id then
    logger:warningf("Incorrect task id, found %s, expected %s\n",
		    taskid, self.id)
    return
  end
  if not self.reduction[key] then
    logger:warningf("Incorrect key %s\n", key)
    return
  end
  if self.reduce_done[key] then
    logger:warningf("Reduce key %s already done\n", key)
    return
  end
  -- mark the key as done
  self.reduce_done[key]  = true
  self.reduce_count      = self.reduce_count + 1
  --
  self.map_reduce_result[key] = result
  -- if any reduce work is pending, give it to the current worker
  if #self.reduction_pending_list > 0 then
    local job_key   = table.remove(self.reduction_pending_list, 1)
    local job_value = self.reduction[job_key]
    self.reduce_worker[key],
    self.reduce_worker[job_key] = nil,self.reduce_worker[key]
    self.reduce_worker[job_key]:do_reduce(self,
					  select_handler,
					  logger,
					  job_key, job_value)
  end
  if self.reduce_count == self.reduce_size then
    self.state = "REDUCE_FINISHED"
  end
  return true
end

function task_methods:do_sequential()
  self.select_handler:send(self.conn,
			   string.format("SEQUENTIAL %s",
					 table.tostring(map_reduce_result)))
  self.select_handler:receive(self.conn,
			      function(conn,msg)
				local msg = table.unpack(string.tokenize(msg or ""))
				if msg ~= "SEQUENTIAL_DONE" then
				  self.state = "ERROR"
				  return "ERROR"
				else
				  self:process_sequential()
				  return "OK"
				end
			      end)
  self.state = "SEQUENTIAL"
end

function task_methods:process_sequential()
  self.state = "SEQUENTIAL_FINISHED"
end

function task_methods:do_loop()
  self.select_handler:send(self.conn,
			   string.format("LOOP %s", self.id))
  self.select_handler:receive(self.conn,
			      function(conn,msg)
				local result = common.load(logger,msg)
				if not result then return "ERROR" end
				task:process_loop(result)
				return "OK"
			      end)
  self.state = "LOOP"
end

function task_methods:process_loop(result)
  if result then self.state = "PREPARED"
  else
    self.select_handler:send(self.conn, "EXIT")
    self.select_handler:receive(self.conn)
    self.select_handler:close(self.conn, function() self.state = "FINISHED" end)
  end
end
