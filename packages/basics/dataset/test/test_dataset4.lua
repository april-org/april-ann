-- vamos a probar matrices 3 dimensionales

-- matrix de talla 4x3x2 viene a ser 2 capas de:
--
--         col1  col2 col3
-- fila1
-- fila2
-- fila3
-- fila4

mat = matrix(4,3,2)

-- un dataset que recorre la matriz por capas de profundidad:
ds = dataset.matrix(mat,{
		      patternSize={4,3,1},
		      offset={0,0,0},
		      stepSize={0,0,1},
		      numSteps={1,1,2},
		    })


ds:putPattern(1,{1,2,3,4,5,6,7,8,9,10,11,12})
-- primera capa:
--  1  2  3
--  4  5  6
--  7  8  9
-- 10 11 12
ds:putPattern(2,{13,14,15,16,17,18,19,20,21,22,23,24})
-- segunda capa:
-- 13 14 15
-- 16 17 18
-- 19 20 21
-- 22 23 24

-- recorremos la matriz pero por filas
dsfilas = dataset.matrix(mat,{
			   patternSize={1,3,2},
			   offset={0,0,0},
			   stepSize={1,0,0},
			   numSteps={4,1,1},
			 })

-- vamos a imprimir cada fila:
for i,pat in dsfilas:patterns() do
  printf("Patron %d -> %s\n",i,table.concat(pat,","))
end

-- salida que pensamos que deberia de dar:
-- Patron 1 -> 1,13,2,14,3,15
-- Patron 2 -> 4,16,5,17,6,18
-- Patron 3 -> 7,19,8,20,9,21
-- Patron 4 -> 10,22,11,23,12,24

-- salida del programa
-- Patron 1 -> 1,13,2,14,4,16
-- Patron 2 -> 4,16,5,17,7,19
-- Patron 3 -> 7,19,8,20,10,22
-- Patron 4 -> 10,22,11,23,12,24 <- curiosamente esta bien
