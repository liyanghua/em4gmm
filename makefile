# Algoritmo EM eficiente para GMM por Juan Daniel Valor Miro bajo la GPLv2.
CFLAGS := -O3
CC := gcc

all:
	$(CC) $(CFLAGS) gmm.c class.c -o gmmclass -lm
	$(CC) $(CFLAGS) gmm.c train.c -o gmmtrain -lm
	
clean:
	rm -f gmmtrain gmmclass
