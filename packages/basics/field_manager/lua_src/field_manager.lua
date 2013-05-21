-- wrapper para field_manager

class("field_manager")

-- constructor
function field_manager:__call()
  local obj = { fields = {} }
  obj = class_instance(obj, self, true)
  return obj
end

-- devuelve un campo del field_manager
function field_manager:get_field(name)
   return self.fields[name]
end

--
function field_manager:load_lines(s)
  if type(s) ~= 'table' or
    type(s.filename) ~= 'string' or type(s.field) ~= 'string' then
    error("Error in field_manager load_lines method incorrect arguments\n")
  end
  local aux = {}
  for line in io.lines(s.filename) do
    table.insert(aux, line)
  end
  self.fields[s.field] = aux
end

--
function field_manager:apply(s)
  -- comprobar argumentos
  if type(s) ~= 'table' or
    type(s.output_fields)~= 'table' or
    type(s.input_fields) ~= 'table' or
    type(s.the_function) ~= 'function' then
    error("Error in field_manager apply method incorrect arguments\n")
  end
  -- comprobar que el campo input_fields tiene al menos un campo
  local num_fields = table.getn(s.input_fields)
  if num_fields == 0 then
    error("Error in field_manager apply method at least one input field is needed\n")
  end
  -- comprobar que todos los campos de input_fields son adecuados
  for i,fieldname in ipairs(s.input_fields) do
    if type(self.fields[fieldname]) ~= 'table' then
      error(string.format("Error in field_manager apply method "..
			  "data_field %s not found incorrect arguments\n",fieldname))
    end
  end
  local num_results = table.getn(s.output_fields)
  -- crear las tablas de resultados si hace falta:
  for j=1,num_results do
    if self.fields[s.output_fields[j]] == nil then
      self.fields[s.output_fields[j]] = {}
    end
  end
  -- iterar sobre los datos
  local size_data = table.getn(self.fields[s.input_fields[1]])
  for i = 1,size_data do -- aplicar funcion al dato i-esimo
    local argumentos = {}
    for j=1,num_fields do
      table.insert(argumentos,self.fields[s.input_fields[j]][i])
    end
    table.insert(argumentos,i)
    local resul = {s.the_function(unpack(argumentos))}
    if table.getn(resul) ~= num_results then
       error(string.format("Error in field_manager apply method at index %d",i))
    end
    -- meter los datos
    for j=1,num_results do
       self.fields[s.output_fields[j]][i] = resul[j]
    end      
  end
end

function field_manager:delete_field(fieldname)
  self.fields[fieldname] = nil
end

