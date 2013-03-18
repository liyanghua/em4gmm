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
		s+=maximum; /* Fast classifier using Viterbi aproximation. */
	}
	return (s*0.5)/feas->samples;
}

/* Initialize the classifier by calculating the non-data dependant part. */
decimal gmm_init_classifier(gmm *gmix){
	decimal cache=gmix->dimension*(-0.5)*log(2*NUM_PI);
	number m,j; gmix->llh=0;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].cgauss=gmix->mix[m]._z=0;
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m].cgauss+=log(gmix->mix[m].dcov[j]);
			gmix->mix[m].dcov[j]=1/gmix->mix[m].dcov[j];
			gmix->mix[m]._mean[j]=gmix->mix[m]._dcov[j]=0; /* Caches to 0. */
		}
		gmix->mix[m].cgauss=2*(gmix->mix[m].prior-((gmix->mix[m].cgauss*0.5)+cache));
	}
}

/* Parallel implementation of the E Step of the EM algorithm. */
void *thread_trainer(void *tdata){
	trainer *info=(trainer*)tdata; /* Get the data for the thread and alloc memory. */
	gmm *gmix=info->gmix; data *feas=info->feas;
	decimal *zval=(decimal*)calloc(2*gmix->num,sizeof(decimal)),*prob=zval+gmix->num;
	decimal *mean=(decimal*)calloc(2*gmix->num*gmix->dimension,sizeof(decimal));
	decimal *dcov=mean+(gmix->num*gmix->dimension),llh=0,x,tz,rmean,maximum;
	number i,j,m,inc;
	for(i=info->ini;i<info->end;i++){
		maximum=-HUGE_VAL;
		for(m=0;m<gmix->num;m++){ /* Compute expected class value of the sample. */
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
		llh+=(rmean=maximum+log(x));
		for(m=0;m<gmix->num;m++){ /* Accumulate counts of the sample in memory. */
			zval[m]+=(tz=exp(prob[m]-rmean)); inc=m*j;
			for(j=0;j<gmix->dimension;j++){
				mean[inc+j]+=(x=tz*feas->data[i][j]);
				dcov[inc+j]+=x*feas->data[i][j];
			}
		}
	}
	pthread_mutex_lock(info->mutex); /* Accumulate counts obtained to the mixture. */
	gmix->llh+=llh;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m]._z+=zval[m]; inc=m*j;
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m]._mean[j]+=mean[inc+j];
			gmix->mix[m]._dcov[j]+=dcov[inc+j];
		}
	}
	pthread_mutex_unlock(info->mutex);
	free(zval); free(mean);
	pthread_exit(NULL);
}

/* Perform one iteration of the EM algorithm with the data and the mixture indicated. */
decimal gmm_EMtrain(data *feas,gmm *gmix){
	pthread_mutex_t *mutex=(pthread_mutex_t*)calloc(1,sizeof(pthread_mutex_t));
	trainer *t=(trainer*)calloc(NUM_THREADS,sizeof(trainer));
	number m,i,j,inc; decimal tz,x;
	/* Calculate expected value and accumulate the counts (E Step). */
	gmm_init_classifier(gmix);
	pthread_mutex_init(mutex,NULL);
	inc=feas->samples/NUM_THREADS;
	for(i=0;i<NUM_THREADS;i++){ /* Set and launch the parallel training. */
		t[i].feas=feas; t[i].gmix=gmix;
		t[i].mutex=mutex; t[i].ini=i*inc;
		t[i].end=(i==NUM_THREADS-1)?(feas->samples):((i+1)*inc);
		pthread_create(&t[i].thread,NULL,thread_trainer,(void*)&t[i]);
	}
	for(i=0;i<NUM_THREADS;i++) /* Wait to the end of the parallel training. */
		pthread_join(t[i].thread,NULL);
	pthread_mutex_destroy(mutex);
	/* Estimate the new parameters of the Gaussian Mixture (M Step). */
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].prior=log((tz=gmix->mix[m]._z)/feas->samples);
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m].mean[j]=(x=gmix->mix[m]._mean[j]/tz);
			gmix->mix[m].dcov[j]=(gmix->mix[m]._dcov[j]/tz)-(x*x);
			if(gmix->mix[m].dcov[j]<gmix->mcov[j]) /* Smoothing covariances. */
				gmix->mix[m].dcov[j]=gmix->mcov[j];
		}
	}
	return gmix->llh/feas->samples;
}

/* Allocate contiguous memory to create a new Gaussian Mixture. */
inline gmm *gmm_create(number n,number d){
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
