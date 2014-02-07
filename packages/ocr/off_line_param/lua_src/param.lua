function ocr.off_line.param.help()

print[[
Paquete ocr.off_line.param
--------------------------

Metodos (de clase):

 - ocr.off_line.param.geom(img, PARAMS_LIST)

   PARAMS_LIST es una cadena que puede contener las siguientes letras:
	S: contorno superior
	I: contorno inferior
	E: energia
	P: posicion media de la energia
	D: desv. tipica de la posicion media de la energia

	T: numero de trazos

	Q: derivada1 del c. sup
	A: derivada1 del c. inf

	Z: derivada2 del c. sup
	X: derivada2 del c. inf

	H: "Altura" -> contorno_inf - contorno_sup
	M: Derivada de la posicion media de la energía

   Si se ponen en minusculas se normalizan respecto a las lineas base
]]


end
