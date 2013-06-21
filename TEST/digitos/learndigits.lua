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
description = "256 inputs 256 tanh 128 tanh 10 log_softmax"
inf         = -0.1
sup         =  0.1
bunch_size  =  tonumber(arg[1]) or 64

thenet  = ann.mlp.all_all.generate(description)
trainer = trainable.supervised_trainer(thenet,
				       ann.loss.multi_class_cross_entropy(10))
trainer:build()
trainer:randomize_weights{
  random = aleat,
  inf    = inf,
  sup    = sup,
}

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
    resul   = trainer:calculate(val_input:getPattern(index)):rewrap(10)
    resul:exp()
    desired = val_output:getPattern(index)
    inputpattern  = train_input:getPattern(index)
    verdigito(inputpattern)
    printf("Labels: ")
    for i=0,9 do printf("    %d",i) end; printf("\n")
    printf("Target: ")
    for i = 1,10 do printf(" %.2f",desired[i]) end; printf("\n")
    printf("Output: ")
    max,imax = resul:max()
    clasificado = imax
    for i = 1,10 do printf(" %.2f",resul:get(i))   end; printf("\n")
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
  shuffle        = aleat,
  bunch_size     = bunch_size,
}

datosvalidar = {
  input_dataset  = val_input,
  output_dataset = val_output,
  bunch_size     = bunch_size,
}

thenet:set_option("learning_rate", 0.04)
thenet:set_option("momentum",      0.02)
thenet:set_option("weight_decay",  1e-05)

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
  print("Epoch Training  Validation")
  for epoch = 1,epocas do
    totalepocas = totalepocas+1
    errortrain  = trainer:train_dataset(datosentrenar)
    errorval    = trainer:validate_dataset(datosvalidar)
    printf("%4d  %.7f %.7f\n",
	   totalepocas,errortrain,errorval)
  end
end

function entrenar_reemplazo(epocas,reemplazos)
  datosentrenar.replacement = reemplazos
  print("Epoch Training  Validation")
  for epoch = 1,epocas do
    totalepocas = totalepocas+1
    errortrain  = trainer:train_dataset(datosentrenar)
    errorval    = trainer:validate_dataset(datosvalidar)
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
	mlp.save(trainer, filename,
		 (tipoAB == "A" and "ascii") or "binary",
		 (tipoOLD == "S" and "old") or nil)
      else
	ann.mlp.all_all.save(trainer, filename, 
			     (tipoAB == "A" and "ascii") or "binary",
			     (tipoOLD == "S" and "old") or nil)
      end
    elseif opc == 5 then
      printf("Cargar la red, nombre: ")
      filename = io.read('*l')
      printf('Leemos la red en "%s"\n',filename)
      print("Antes",trainer)
      if not use_adrian_mlp then
	trainer = mlp.load(filename)
      else
	trainer = ann.mlp.all_all.load(filename, bunch_size)
      end
      print("Despues",trainer)
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
      trainer:classify{
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
      trainer:classify{
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

