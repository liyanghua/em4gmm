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
#include "gmm.h"

/* Allocate contiguous memory to create a new Gaussian Mixture. */
gmm *gmm_create(number n,number d){
	gmm *gmix=(gmm*)calloc(1,sizeof(gmm));
	gmix->mcov=(decimal*)calloc(gmix->dimension=d,sizeof(decimal));
	gmix->mix=(gauss*)calloc(gmix->num=n,sizeof(gauss));
	for(n=0;n<gmix->num;n++){
		gmix->mix[n].mean=(decimal*)calloc(gmix->dimension*4,sizeof(decimal));
		gmix->mix[n].dcov=gmix->mix[n].mean+gmix->dimension;
		gmix->mix[n]._mean=gmix->mix[n].dcov+gmix->dimension;
		gmix->mix[n]._dcov=gmix->mix[n]._mean+gmix->dimension;
	}
	return gmix;
}

/* Create and initialize the Mixture with maximum likelihood and disturb the means. */
gmm *gmm_initialize(data *feas,number nmix){
	gmm *gmix=gmm_create(nmix,feas->dimension);
	number i,j,k,b=feas->samples/gmix->num,bc=0;
	decimal x=1.0/gmix->num;
	/* Initialize the first Gaussian with maximum likelihood. */
	gmix->mix[0].prior=log(x);
	for(j=0;j<gmix->dimension;j++){
		gmix->mix[0].dcov[j]=x*feas->variance[j];
		gmix->mcov[j]=0.001*gmix->mix[0].dcov[j];
	}
	/* Disturb all the means creating C blocks of samples. */
	for(i=gmix->num-1;i>=0;i--,bc+=b){
		gmix->mix[i].prior=gmix->mix[0].prior;
		for(j=bc,x=0;j<bc+b;j++) /* Compute the mean of a group of samples. */
			for(k=0;k<gmix->dimension;k++)
				gmix->mix[i]._mean[k]+=feas->data[j][k];
		for(k=0;k<gmix->dimension;k++){ /* Disturbe the sample mean for each mixture. */
			gmix->mix[i].mean[k]=(feas->mean[k]*0.9)+(0.1*gmix->mix[i]._mean[k]/b);
			gmix->mix[i].dcov[k]=gmix->mix[0].dcov[k];
		}
	}
	return gmix;
}

/* Load the Gaussian Mixture from the file received as parameter. */
gmm *gmm_load(char *filename){
	number m,d;
	FILE *f=fopen(filename,"rb");
	if(!f)fprintf(stderr,"Error: Not %s model file found.\n",filename),exit(1);
	fread(&d,1,sizeof(number),f);
	fread(&m,1,sizeof(number),f);
	gmm *gmix=gmm_create(m,d);
	fread(gmix->mcov,sizeof(decimal)*gmix->dimension,1,f);
	for(m=0;m<gmix->num;m++){
		fread(&gmix->mix[m].prior,sizeof(decimal),1,f);
		fread(&gmix->mix[m].cgauss,sizeof(decimal),1,f);
		fread(gmix->mix[m].mean,gmix->dimension*sizeof(decimal)*2,1,f);
	}
	fclose(f);
	return gmix;
}

/* Save the Gaussian Mixture to the file received as parameter. */
void gmm_save(char *filename,gmm *gmix){
	number m;
	FILE *f=fopen(filename,"wb");
	if(!f)fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
	fwrite(&gmix->dimension,1,sizeof(number),f);
	fwrite(&gmix->num,1,sizeof(number),f);
	fwrite(gmix->mcov,sizeof(decimal)*gmix->dimension,1,f);
	for(m=0;m<gmix->num;m++){
		fwrite(&gmix->mix[m].prior,sizeof(decimal),1,f);
		fwrite(&gmix->mix[m].cgauss,sizeof(decimal),1,f);
		fwrite(gmix->mix[m].mean,sizeof(decimal)*gmix->dimension*2,1,f);
	}
	fclose(f);
}

/* Free the allocated memory of the Gaussian Mixture. */
void gmm_delete(gmm *gmix){
	number i;
	for(i=0;i<gmix->num;i++)
		free(gmix->mix[i].mean);
	free(gmix->mix);
	free(gmix);
}

/* Save the classifier log as a jSON file. */
void gmm_results_save(char *filename,cluster *c){
	number i,j; gzFile f=gzopen(filename,"wb");
	if(!f)fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
	gzprintf(f,"{\n\t\"global_score\": %.10f,",c->result);
	gzprintf(f,"\n\t\"samples\": %i,\n\t\"classes\": %i,",c->samples,c->mixtures);
	gzprintf(f,"\n\t\"class_occupation\": [ %i",c->freq[0]);
	for(i=1;i<c->mixtures;i++)
		gzprintf(f,", %i",c->freq[i]);
	gzprintf(f," ],\n\t\"samples_results\": [ ");
	for(i=0;i<c->samples;i++){
		gzprintf(f,"\n\t\t { \"sample\": %i, \"class\": %i, ",i,c->mix[i]);
		gzprintf(f,"\"lprob\": [ %.10f",c->prob[i*c->mixtures]);
		for(j=1;j<c->mixtures;j++)
			gzprintf(f,", %.10f",c->prob[i*c->mixtures+j]);
		if(i==c->samples-1) gzprintf(f," ] }");
		else gzprintf(f," ] },");
	}
	gzprintf(f,"\n\t]\n}");
	gzclose(f);
}

/* Free the allocated memory of the classifier results. */
void gmm_results_delete(cluster *c){
	free(c->mix);
	free(c->freq);
	free(c->prob);
	free(c);
}
