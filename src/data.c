/* Expectation Maximization for Gaussian Mixture Models.
Copyright (C) 2012-2013 Juan Daniel Valor Miro

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. */

#include "global.h"
#include "data.h"

/* Load the samples from a plain text file with the specified format. */
data *feas_load(char *filename){
	data *feas=(data*)calloc(1,sizeof(data));
	number i,j;
	char header[9];
	FILE *f=fopen(filename,"r");
	if(!f) fprintf(stderr,"Error: Not %s feature file found.\n",filename),exit(1);
	fscanf(f,"%s",header);
	if(strcmp(header,"AKREALTF")!=0)
		fprintf(stderr,"Error: Wrong %s feature file format.\n",filename),exit(1);
	fscanf(f,"%i",&feas->dimension);
	fscanf(f,"%i",&feas->samples);
	feas->data=(decimal**)malloc(feas->samples*sizeof(decimal*));
	decimal *aux=(decimal*)malloc(feas->dimension*feas->samples*sizeof(decimal));
	for(i=0;i<feas->samples;i++){
		feas->data[i]=aux;
		aux+=feas->dimension;
		for(j=0;j<feas->dimension;j++)
			fscanf(f,"%lf",&feas->data[i][j]);
	}
	fclose(f);
	return feas;
}

/* Free the allocated memory of the samples. */
void feas_delete(data *feas){
	free(feas->data[0]);
	free(feas->data);
	free(feas);
}
