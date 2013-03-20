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
DFLAGS := -g
CC := gcc
BINDIR := /usr/bin

SRC := src
BIN := bin
DAT := dat
ALL := $(SRC)/data.c $(SRC)/gmm.c

all:
	$(CC) $(CFLAGS) -fprofile-generate $(ALL) $(SRC)/train.c -o $(BIN)/gmmtrain -pthread -lz -lm
	strip $(BIN)/gmmtrain
	chmod +x $(BIN)/gmmtrain
	$(BIN)/gmmtrain 128 $(DAT)/data.gz $(DAT)/data.gmm
	$(CC) $(CFLAGS) -fprofile-use $(ALL) $(SRC)/train.c -o $(BIN)/gmmtrain -pthread -lz -lm
	strip $(BIN)/gmmtrain
	chmod +x $(BIN)/gmmtrain
	rm -f *.gcda
	$(CC) $(CFLAGS) -fprofile-generate $(ALL) $(SRC)/class.c -o $(BIN)/gmmclass -pthread -lz -lm
	strip $(BIN)/gmmclass
	chmod +x $(BIN)/gmmclass
	$(BIN)/gmmclass $(DAT)/data.gz $(DAT)/data.gmm $(DAT)/data.gmm
	$(CC) $(CFLAGS) -fprofile-use $(ALL) $(SRC)/class.c -o $(BIN)/gmmclass -pthread -lz -lm
	strip $(BIN)/gmmclass
	chmod +x $(BIN)/gmmclass
	rm -f *.gcda

debug:
	$(CC) $(DFLAGS) $(ALL) $(SRC)/class.c -o $(BIN)/gmmclass -pthread -lz -lm
	$(CC) $(DFLAGS) $(ALL) $(SRC)/train.c -o $(BIN)/gmmtrain -pthread -lz -lm
	chmod +x $(BIN)/gmmclass
	chmod +x $(BIN)/gmmtrain

install:
	cp -f $(BIN)/gmmclass $(BINDIR)/gmmclass
	cp -f $(BIN)/gmmtrain $(BINDIR)/gmmtrain
	chmod +x $(BINDIR)/gmmclass
	chmod +x $(BINDIR)/gmmtrain

clean:
	rm -f $(BIN)/gmmtrain
	rm -f $(BIN)/gmmtrain
	rm -f $(DAT)/data.gmm
