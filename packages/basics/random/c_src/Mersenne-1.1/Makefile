# Makefile for Wagner's Mersenne Twister class (MersenneTwister.h)

# Define compiler and compilation options

CC = g++
CFLAGS = -ansi -Wall
COPTIM = -O3


# Compile example program

.PHONY: all run test compare clean package

all: example
	@ echo "Run the example program with './example'"
	@ echo "Or use 'make test' to test the output and speed of MersenneTwister.h"

example: example.cpp MersenneTwister.h
	$(CC) $(CFLAGS) -o example example.cpp

run: example
	./example


# Compile test suite

test:
	@ $(MAKE) -C Test/ CC=$(CC) CFLAGS=$(CFLAGS) COPTIM=$(COPTIM)


# Clean away executable and output files

clean:
	@ rm -f example state.dat core
	@ $(MAKE) -C Test/ clean


# Package files for distribution

DIR = Mersenne-1.1

package: clean
	rm -rf $(DIR)/
	mkdir $(DIR)
	cp README $(DIR)/
	cp MersenneTwister.h $(DIR)/
	cp example.cpp $(DIR)/
	cp Makefile $(DIR)/
	cp -R Test $(DIR)/
	cat MersenneTwister.html | sed '/cutforpkg/,/cutforpkg/ s/.*//' > $(DIR)/MersenneTwister.html
	cp Random.jpg $(DIR)/
	tar -zcvf $(DIR).tar.gz $(DIR)/
	zip -r $(DIR).zip $(DIR)/
