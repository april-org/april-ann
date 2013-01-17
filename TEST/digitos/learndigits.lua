use_adrian_mlp = (type(ann) == "table")

function verdigito(digito)
  local out = "------------------\n"
  for x = 0,15 do
    out = out..'|'
    for y = 1,16 do
      out = out..(((digito[16*x+y] == 1) and "#") or " ")
    end
    out = out..'|\n'
  end
  out = out.."------------------\n"
  printf(out)
end

m1 = matrix.loadImage("digits.png", "gray")
print(table.concat(m1:dim(), " "))
train_input = dataset.matrix(m1,
			     {
			       patternSize = {16,16},
			       offset      = {0,0},
			       numSteps    = {80,10},
			       stepSize    = {16,16},
			       orderStep   = {1,0}
			     })

val_input  = dataset.matrix(m1,
			    {
			      patternSize = {16,16},
			      offset      = {1280,0},
			      numSteps    = {20,10},
			      stepSize    = {16,16},
			      orderStep   = {1,0}
			    })

-- una matriz pequenya la podemos cargar directamente
m2 = matrix(10,{1,0,0,0,0,0,0,0,0,0})

-- ojito con este dataset, fijaros que usa una matriz de dim 1 y talla
-- 10 PERO avanza con valor -1 y la considera CIRCULAR en su unica
-- dimension

train_output = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {800},
				stepSize    = {-1},
				circular    = {true}
			      })

val_output   = dataset.matrix(m2,
			      {
				patternSize = {10},
				offset      = {0},
				numSteps    = {200},
				stepSize    = {-1},
				circular    = {true}
			      })

-- esto es una prueba
--doble_train_input  = dataset.union{train_input,train_input}
--doble_train_output = dataset.union{train_output,train_output}

-- un generador de valores aleatorios... y otros parametros
semilla     = 1234
aleat       = random(semilla)
description = "256 inputs 256 tanh 128 tanh 10 softmax"
inf         = -0.1
sup         =  0.1
bunch_size  =  tonumber(arg[1]) or 64

if use_adrian_mlp then
  lared = ann.mlp.all_all.generate{
    bunch_size  = bunch_size,
    topology    = description,
    random      = aleat,
    inf         = inf,
    sup         = sup,
  }
  ann.mlp.all_all.save(lared, "new.net", "ascii", "old")
  --lared:set_error_function(ann.error_functions.cross_entropy())
else
  lared = mlp.generate_with_bunch{
    bunch_size = bunch_size,
    topology   = description,
    random     = aleat,
    inf        = inf,
    sup        = sup
  }
  mlp.save(lared, "old.net", "ascii", "old")
end

-- otro para mostrar datos de validacion
otrorand = random(5678)
-- algunas funciones
function ver_resultado()
  --lared:prune_output_neurons()
  repeat
    --clrscr()
    digito  = otrorand:randInt(0,9)
    autor   = otrorand:randInt(0,19)
    index   = 1+autor*10+digito
    if use_adrian_mlp then
      resul   = lared:calculate(val_input:getPattern(index))
    else
      resul   = lared:use(val_input:getPattern(index))
    end
    desired = val_output:getPattern(index)
    inputpattern  = train_input:getPattern(index)
    verdigito(inputpattern)
    printf("Labels: ")
    for i=0,9 do printf("    %d",i) end; printf("\n")
    printf("Target: ")
    for i = 1,10 do printf(" %.2f",desired[i]) end; printf("\n")
    printf("Output: ")
    max = resul[1]
    imax = 1
    for i= 2,10 do if resul[i]>max then max = resul[i] imax = i end end
    clasificado = imax-1
    for i = 1,10 do printf(" %.2f",resul[i])   end; printf("\n")
    printf("Autor %2d Digito %d\n",autor,digito)
    printf("MLP clasifica   %d %s\n",clasificado,clasificado == digito and ":)" or ":'(")
    --    printf("\n\nLos valores de las neuronas de las capas ocultas:\n%s\n\n",
    --	   string.join(lared:activation_values(),","))
    print("Pulse intro para continuar, una letra e intro para parar")
    a = io.read('*l')
  until string.len(a) > 0
end


-- datos para entrenar

datosentrenar = {
  input_dataset  = train_input,
  output_dataset = train_output,
  shuffle        = aleat
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
}

learning_rate = 0.01
momentum      = 0.00
weight_decay  = 0.00

if not use_adrian_mlp then
  datosentrenar.learning_rate  = learning_rate
  datosentrenar.momentum       = momentum
  datosentrenar.weight_decay   = weight_decay
else
  lared:set_option("learning_rate", learning_rate)
  lared:set_option("momentum",      momentum)
  lared:set_option("weight_decay",  weight_decay)
end

-- datos para guardar una matriz de salida de la red
msalida = matrix(200,10)
dsalida = dataset.matrix(msalida, {
			   patternSize={1,10},offset={0,0},numSteps={200,1},stepSize={1,0}})
datosusar = {
  input_dataset  = val_input,
  output_dataset = dsalida,
}

totalepocas = 0

function entrenar(epocas)
  lared:set_option("learning_rate", learning_rate)
  lared:set_option("momentum",      momentum)
  lared:set_option("weight_decay",  weight_decay)

  print("Epoch Training  Validation")
  for epoch = 1,epocas do
    totalepocas = totalepocas+1
    if use_adrian_mlp then
      errortrain = lared:train_dataset(datosentrenar)
      errorval   = lared:validate_dataset(datosvalidar)
    else
      errortrain = lared:train(datosentrenar)
      errorval   = lared:validate(datosvalidar)
    end
    printf("%4d  %.7f %.7f\n",
	   totalepocas,errortrain,errorval)
  end
end

function entrenar_reemplazo(epocas,reemplazos)
  datosentrenar.replacement = reemplazos
  print("Epoch Training  Validation")
  for epoch = 1,epocas do
    totalepocas = totalepocas+1
    if use_adrian_mlp then
      errortrain = lared:train_dataset(datosentrenar)
      errorval   = lared:validate_dataset(datosvalidar)
    else
      errortrain = lared:train(datosentrenar)
      errorval   = lared:validate(datosvalidar)
    end
    printf("%4d  %.7f %.7f\n",
	   totalepocas,errortrain,errorval)
  end
  datosentrenar.replacement = nil
end

function menu()
  --clrscr()
  printf("-------- ENTRENAMIENTO DIGITOS -------- \n")
  printf(" 1 - Entrenar epocas\n")
  printf(" 2 - Entrenar con reemplazo\n")
  printf(" 3 - Ver resultados\n")
  printf(" 4 - Salvar red\n")
  printf(" 5 - Cargar red\n")
  printf(" 6 - Ver datos\n")
  printf(" 7 - Clasificar\n")
  printf(" 8 - Salir\n")
  printf("Opcion:")
  opc = tonumber(io.read('*l'))
  return opc
end

function ppal()
  repeat
    local opc = menu()
    if opc == 1 then
      printf("cuantas epocas? ")
      cuantas = tonumber(io.read('*l'))
      entrenar(cuantas)
      print("Pulse intro para continuar")
      io.read('*l')
    elseif opc == 2 then
      printf("cuantos reemplazos? ")
      reemplazos = tonumber(io.read('*l'))
      printf("cuantas epocas? ")
      cuantas = tonumber(io.read('*l'))
      entrenar_reemplazo(cuantas,reemplazos)
      print("Pulse intro para continuar")
      io.read('*l')
    elseif opc == 3 then
      ver_resultado()
    elseif opc == 4 then
      printf("Salvar la red, nombre: ")
      filename = io.read('*l')
      printf("Ascii o binario? (A/B): ")
      tipoAB = io.read('*l')
      printf("Guardamos pesos iteracion anterior? (S/N): ")
      tipoOLD = io.read('*l')
      if not use_adrian_mlp then
	mlp.save(lared, filename,
		 (tipoAB == "A" and "ascii") or "binary",
		 (tipoOLD == "S" and "old") or nil)
      else
	ann.mlp.all_all.save(lared, filename, 
			     (tipoAB == "A" and "ascii") or "binary",
			     (tipoOLD == "S" and "old") or nil)
      end
    elseif opc == 5 then
      printf("Cargar la red, nombre: ")
      filename = io.read('*l')
      printf('Leemos la red en "%s"\n',filename)
      print("Antes",lared)
      if not use_adrian_mlp then
	lared = mlp.load(filename)
      else
	lared = ann.mlp.all_all.load(filename, bunch_size)
      end
      print("Despues",lared)
      print("Pulse intro para continuar")
      io.read('*l')
    elseif opc == 6 then
      for index=1,train_output:numPatterns() do
	--clrscr()
	print"Introduce cualquier cosa ademas de enter parar parar"
	verdigito(train_input:getPattern(index))
	print(table.concat(train_output:getPattern(index),","))
	a = io.read('*l')
	if (string.len(a) > 0) then break end
      end
    elseif opc == 7 then
      -- clasificamos
      -------------------------------------
      local aux_m = matrix(train_input:numPatterns())
      local aux_ds = dataset.matrix(aux_m)
      lared:classify{
	input_dataset  = train_input,
	output_dataset = aux_ds,
      }
      local errors = 0
      for i=1,aux_ds:numPatterns() do
	local pat = train_output:getPattern(i)
	v,p = table.max(pat)
	if p ~= aux_ds:getPattern(i)[1] then
	  errors = errors + 1
	end
      end
      print ("ERROR EN TRAINING: ",
	     errors/aux_ds:numPatterns() .. "%", errors)
      -------------------------------------
      local aux_m = matrix(val_input:numPatterns())
      local aux_ds = dataset.matrix(aux_m)
      lared:classify{
	input_dataset  = val_input,
	output_dataset = aux_ds,
      }
      local errors = 0
      for i=1,aux_ds:numPatterns() do
	local pat = val_output:getPattern(i)
	v,p = table.max(pat)
	if p ~= aux_ds:getPattern(i)[1] then
	  errors = errors + 1
	end
      end
      print ("ERROR EN VALIDATION: ",
	     errors/aux_ds:numPatterns() .. " %", errors)
      print("Pulse intro para continuar")
      io.read('*l')
    end
    
  until opc == 8
end

printf("He cargado\n")
ppal()

