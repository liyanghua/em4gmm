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

/* Efficient Gaussian Mixture classifier using a Viterbi aproximation. */
decimal gmm_classify(data *feas,gmm *gmix){
	decimal x,maximum,prob,s=0;
	number i,m,j;
	for(i=0;i<feas->samples;i++){
		maximum=-HUGE_VAL;
		for(m=0;m<gmix->num;m++){
			prob=gmix->mix[m].cgauss; /* The non-data dependant part was precalculated. */
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob-=(x*x)*gmix->mix[m].dcov[j];
			}
			if(maximum<prob)maximum=prob;
		}
		s+=maximum;
	}
	return (s*0.5)/feas->samples;
}

/* Initialize the classifier by calculating the non-data dependant part. */
decimal gmm_init_classifier(gmm *gmix){
	decimal cache=gmix->dimension*(-0.5)*log(2*M_PI);
	number m,j;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].cgauss=0;
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m].cgauss+=log(gmix->mix[m].dcov[j]);
			gmix->mix[m].dcov[j]=1/gmix->mix[m].dcov[j];
			gmix->mix[m]._mean[j]=gmix->mix[m]._dcov[j]=0; /* Caches to 0. */
		}
		gmix->mix[m].cgauss=2*(gmix->mix[m].prior-((gmix->mix[m].cgauss*0.5)+cache));
	}
}

/* Perform one iteration of the EM algorithm with the data and the mixture indicated. */
decimal gmm_EMtrain(data *feas,gmm *gmix){
	decimal *prob=(decimal*)calloc(gmix->num<<1,sizeof(decimal));
	decimal tz,mean,llh=0,*z=prob+gmix->num,x,maximum;
	number i,m,j;
	/* Calculate expected value and accumulate the counts (E Step). */
	gmm_init_classifier(gmix);
	for(i=0;i<feas->samples;i++){
		maximum=-HUGE_VAL;
		for(m=0;m<gmix->num;m++){
			prob[m]=gmix->mix[m].cgauss;
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob[m]-=(x*x)*gmix->mix[m].dcov[j];
			}
			prob[m]*=0.5;
			maximum=maximum>prob[m]?maximum:prob[m];
		}
		for(m=0,x=0;m<gmix->num;m++) /* Do not use Viterbi aproximation. */
			x+=exp(prob[m]-maximum);
		llh+=(mean=maximum+log(x));
		for(m=0;m<gmix->num;m++){
			z[m]+=(tz=exp(prob[m]-mean));
			for(j=0;j<feas->dimension;j++){ /* Accumulate counts. */
				gmix->mix[m]._mean[j]+=(x=tz*feas->data[i][j]);
				gmix->mix[m]._dcov[j]+=x*feas->data[i][j];
			}
		}
	}
	/* Estimate the new parameters of the Gaussian Mixture (M Step). */
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].prior=log(z[m]/feas->samples);
		for(j=0;j<feas->dimension;j++){
			gmix->mix[m].mean[j]=(x=gmix->mix[m]._mean[j]/z[m]);
			gmix->mix[m].dcov[j]=(gmix->mix[m]._dcov[j]/z[m])-(x*x);
			if(gmix->mix[m].dcov[j]<gmix->mcov[j]) /* Smoothing covariances. */
				gmix->mix[m].dcov[j]=gmix->mcov[j];
		}
	}
	free(prob);
	return llh/feas->samples;
}

/* Allocate contiguous memory to create a new Gaussian Mixture. */
inline gmm *gmm_create(number n,number d){
	gmm *gmix=(gmm*)calloc(1,sizeof(gmm));
	gmix->dimension=d;
	gmix->mcov=(decimal*)calloc(gmix->dimension,sizeof(decimal));
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
	decimal x=1.0/gmix->num,y;
	number i,j,b=feas->samples/gmix->num,bc=0,k;
	/* Initialize the first Gaussian with maximum likelihood. */
	gmix->mix[0].prior=log(x);
	for(j=0;j<feas->dimension;j++){
		for(i=0;i<feas->samples;i++){
			gmix->mix[0].mean[j]+=feas->data[i][j];
			gmix->mix[0].dcov[j]+=feas->data[i][j]*feas->data[i][j];
		}
		gmix->mix[0].mean[j]/=feas->samples;
		gmix->mix[0].dcov[j]=x*((gmix->mix[0].dcov[j]/feas->samples)
			-(gmix->mix[0].mean[j]*gmix->mix[0].mean[j]));
		gmix->mcov[j]=0.001*gmix->mix[0].dcov[j];
	}
	/* Disturb all the means creating C blocks of samples. */
	for(i=gmix->num-1;i>=0;i--,bc+=b){
		gmix->mix[i].prior=gmix->mix[0].prior;
		for(j=bc,x=0;j<bc+b;j++)
			for(k=0;k<feas->dimension;k++)
				gmix->mix[i]._mean[k]+=feas->data[j][k];
		for(k=0;k<feas->dimension;k++){
			gmix->mix[i].mean[k]=(gmix->mix[0].mean[k]*0.9)+(0.1*gmix->mix[i]._mean[k]/b);
			gmix->mix[i].dcov[k]=gmix->mix[0].dcov[k];
		}
	}
	return gmix;
}

/* Load the Gaussian Mixture from the file received as parameter. */
gmm *gmm_load(char *filename){
	number m,d;
	FILE *f=fopen(filename,"rb");
	if(!f) fprintf(stderr,"Error: Not %s model file found.\n",filename),exit(1);
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
	if(!f) fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
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
