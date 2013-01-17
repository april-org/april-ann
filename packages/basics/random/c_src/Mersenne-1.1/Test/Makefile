# Makefile for testing various Mersenne Twister implementations

# Define compiler and compilation options

CC = g++
CFLAGS = -ansi -Wall
COPTIM = -O3


# Compile and run test programs

.PHONY: all run test clean

all: test

run: test

WagnerTest: Wagner/Wagner.cpp ../MersenneTwister.h
	$(CC) $(CFLAGS) $(COPTIM) -o WagnerTest -I../ Wagner/Wagner.cpp

BedauxTest: Bedaux/Bedaux.cpp Bedaux/mtrand.h Bedaux/mtrand.cpp
	-$(CC) $(COPTIM) -o BedauxTest Bedaux/Bedaux.cpp Bedaux/mtrand.cpp

BlevinsTest: Blevins/Blevins.cpp Blevins/mt.h Blevins/mt.cc
	-$(CC) $(COPTIM) -o BlevinsTest Blevins/Blevins.cpp Blevins/mt.cc

CokusTest: Cokus/Cokus.cpp Cokus/mt19937ar-cok.c
	-$(CC) $(COPTIM) -o CokusTest Cokus/Cokus.cpp

KuenningTest: Kuenning/Kuenning.cpp Kuenning/mtwist.h Kuenning/mtwist.c
	-$(CC) $(COPTIM) -o KuenningTest Kuenning/Kuenning.cpp Kuenning/mtwist.c

LaddTest: Ladd/Ladd.cpp Ladd/mtprng.h Ladd/mtprng.cpp
	-$(CC) $(COPTIM) -o LaddTest Ladd/Ladd.cpp Ladd/mtprng.cpp

OriginalTest: Original/Original.cpp Original/mt19937ar.c
	-$(CC) $(COPTIM) -o OriginalTest Original/Original.cpp

StandardTest: Standard/Standard.cpp
	-$(CC) $(COPTIM) -o StandardTest Standard/Standard.cpp

YangTest: Yang/Yang.cpp Yang/mtrnd.h
	-$(CC) $(COPTIM) -o YangTest Yang/Yang.cpp

test: WagnerTest BedauxTest BlevinsTest CokusTest KuenningTest LaddTest OriginalTest StandardTest YangTest
	@ echo
	@ echo "Testing output and speed of random number generators, please be patient..."
	./WagnerTest > Wagner.out
	-./BedauxTest > Bedaux.out
	-./BlevinsTest > Blevins.out
	-./CokusTest > Cokus.out
	-./KuenningTest > Kuenning.out
	-./LaddTest > Ladd.out
	-./OriginalTest > Original.out
	-./StandardTest > Standard.out
	-./YangTest > Yang.out
	./Results.sh


# Clean away executable and output files

clean:
	@ rm -f Wagner.out Bedaux.out Blevins.out Cokus.out Kuenning.out Ladd.out Original.out Standard.out Yang.out
	@ rm -f WagnerTest BedauxTest BlevinsTest CokusTest KuenningTest LaddTest OriginalTest StandardTest YangTest
	@ rm -f state.dat *.tmp core
