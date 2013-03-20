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
	char *buff=(char*)calloc(SIZE_BUFFER,sizeof(char));
	number i,r,s=0,d=0,c=0,header=2,point=0,sign=1;
	decimal *aux,next=0,dec=1;
	FILE *f=fopen(filename,"r");
	if(!f) fprintf(stderr,"Error: Not %s feature file found.\n",filename),exit(1);
	while(!feof(f)){
		r=fread(buff,sizeof(char),SIZE_BUFFER,f);
		for(i=0;i<r;i++){
			if(buff[i]>='0'){ /* Convert an array into a decimal number. */
				next=(point==0)?((10*next)+buff[i]-'0'):(next+((dec*=0.1)*(buff[i]-'0'))),c++;
			}else if(buff[i]<'-'&&c>0){
				if(header==0){
					if(d==0){ /* If we start a new sample, set the memory. */
						feas->data[s]=aux;
						aux+=feas->dimension;
					}
					feas->data[s][d]=next*sign; /* Asign the decimal to the sample. */
					feas->mean[d]+=feas->data[s][d];
					feas->variance[d]+=feas->data[s][d]*feas->data[s][d];
					if((++d)==feas->dimension)d=0,s++;
				}else if(header==1){
					feas->samples=(int)next;
					feas->mean=(decimal*)calloc(2*feas->dimension,sizeof(decimal*));
					feas->variance=feas->mean+feas->dimension;
					feas->data=(decimal**)calloc(feas->samples,sizeof(decimal*));
					aux=(decimal*)calloc(feas->dimension*feas->samples,sizeof(decimal));
					header=s=d=0;
				}else if(header==2)feas->dimension=(int)next,header=1;
				sign=dec=1,next=point=c=0;
			}else if(buff[i]=='-')sign=-1;
			else if(buff[i]=='.')point=1;
		}
	}
	free(buff);
	fclose(f);
	for(i=0;i<feas->dimension;i++){ /* Compute the mean and variance of the data. */
		feas->mean[i]/=feas->samples;
		feas->variance[i]=(feas->variance[i]/feas->samples)-(feas->mean[i]*feas->mean[i]);
	}
	return feas;
}

/* Free the allocated memory of the samples. */
void feas_delete(data *feas){
	free(feas->data[0]);
	free(feas->data);
	free(feas->mean);
	free(feas);
}
