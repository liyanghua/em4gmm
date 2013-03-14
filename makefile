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

all:
	$(CC) $(CFLAGS) gmm.c class.c -o gmmclass -lm
	$(CC) $(CFLAGS) gmm.c train.c -o gmmtrain -lm
	strip gmmclass
	strip gmmtrain
	chmod +x gmmclass
	chmod +x gmmtrain

install:
	cp -f gmmclass $(BINDIR)/gmmclass
	cp -f gmmtrain $(BINDIR)/gmmtrain
	chmod +x $(BINDIR)/gmmclass
	chmod +x $(BINDIR)/gmmtrain

clean:
	rm -f gmmtrain
	rm -f gmmclass
