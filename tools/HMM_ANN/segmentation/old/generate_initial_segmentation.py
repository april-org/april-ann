# lee un fichero con este formato:
# 08730_533_10850:271_f2_unknown 34 uu 0 (164,0) (165,1) (166,1) (167,1) (168,0)
# 08730_533_10850:271_f2_unknown 26 nn 3 (124,0) (125,1) (126,1) (127,1) (128,0)
# 08730_533_10850:271_f2_unknown 4 pause 6 (16,0) (17,17) (18,15) (19,26) (20,0)
# 08730_533_10850:271_f2_unknown 14 ch 64 (64,0) (65,7) (66,3) (67,6) (68,0)
# 08730_533_10850:271_f2_unknown 10 an 80 (44,0) (45,4) (46,1) (47,3) (48,0)
# 08730_533_10850:271_f2_unknown 12 bb 88 (54,0) (55,2) (56,1) (57,3) (58,0)
#
# cada linea es de la forma:
# fichero indice fonema primeratrama segmentacion dentro del fonema
#
# la segmentacion dentro del fonema es una lista de (estadoHMM,numero
# de tramas) si sumamos los numero de tramas de todos los estados
# vemos que concuerda con el indice de trama del siguiente fonema
#
# el script genera una matriz unidimensional donde cada elemento es el
# indice del tipo de emision de HMM

from random import randint

phone_list = open("phone_list.txt").read().split()
print(phone_list)

file_list = open("train.txt").read().split()

for filename in file_list:
  print(filename)
  output_sequence = []
  for linea in open("/labo/data/Media/SEG_CI/"+filename+".seg").readlines():
    aux       = linea.split()
    phone     = phone_list.index(aux[2])
    ristra    = [int(x.split(',')[1].replace(")","")) for x in aux[5:-1]]
    numstates = len(ristra)
    numframes = sum(ristra)
    # print phone,ristra
    if numstates > 3:
      ristra[2] += sum(ristra[3:])
      del ristra[3:]
    if numstates < 3:
      ristra = [0,0,0]
      cociente = numframes / 3 # in python3 use //
      resto    = numframes % 3
      for i in range(3):
        ristra[i] = cociente
      for i in range(resto):
          ristra[randint(0,2)] += 1
    st = 3*phone
    for i in range(3):
      for j in range(ristra[i]):
        output_sequence.append(st+i)
  f = open("initial_segmentation/"+filename+".sgm","w")
  f.write("%d\nascii\n" % (len(output_sequence),))
  for i in output_sequence:
    f.write("%d\n" % (i,))
  f.close()

    
