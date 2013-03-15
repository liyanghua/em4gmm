# Expectation Maximization for Gaussian Mixture Models.
# Copyright (C) 2012-2013 Juan Daniel Valor Miro
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

CFLAGS := -O3
CC := gcc
BINDIR := /usr/bin

SRC := src
BIN := bin
ALL := $(SRC)/data.c $(SRC)/gmm.c

all:
	$(CC) $(CFLAGS) $(ALL) $(SRC)/class.c -o $(BIN)/gmmclass -lm
	$(CC) $(CFLAGS) $(ALL) $(SRC)/train.c -o $(BIN)/gmmtrain -lm
	strip $(BIN)/gmmclass
	strip $(BIN)/gmmtrain
	chmod +x $(BIN)/gmmclass
	chmod +x $(BIN)/gmmtrain

install:
	cp -f $(BIN)/gmmclass $(BINDIR)/gmmclass
	cp -f $(BIN)/gmmtrain $(BINDIR)/gmmtrain
	chmod +x $(BINDIR)/gmmclass
	chmod +x $(BINDIR)/gmmtrain

clean:
	rm -f $(BIN)/gmmtrain
	rm -f $(BIN)/gmmclass
