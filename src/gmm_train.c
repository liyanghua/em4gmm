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

typedef struct{
	pthread_t thread;       /* pthread identifier of the thread.   */
	pthread_mutex_t *mutex; /* Common mutex to lock shared data.   */
	data *feas;             /* Shared pointer to loaded samples.   */
	gmm *gmix;              /* Shared pointer to gaussian mixture. */
	number ini, end;        /* Initial and final sample processed. */
}trainer;

/* Parallel implementation of the E Step of the EM algorithm. */
void *thread_trainer(void *tdata){
	trainer *t=(trainer*)tdata; /* Get the data for the thread and alloc memory. */
	gmm *gmix=t->gmix; data *feas=t->feas;
	decimal *zval=(decimal*)calloc(2*gmix->num,sizeof(decimal)),*prob=zval+gmix->num;
	decimal *mean=(decimal*)calloc(2*gmix->num*gmix->dimension,sizeof(decimal));
	decimal *dcov=mean+(gmix->num*gmix->dimension),llh=0,x,tz,rmean,maximum,tmpd;
	number *tfreq=(number*)calloc(gmix->num,sizeof(number));
	number i,j,m,inc,c; decimal mepsilon=log(DBL_EPSILON);
	for(i=t->ini;i<t->end;i++){
		maximum=-HUGE_VAL,c=-1;
		for(m=0;m<gmix->num;m++){ /* Compute expected class value of the sample. */
			prob[m]=gmix->mix[m].cgauss;
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob[m]-=(x*x)*gmix->mix[m].dcov[j];
			}
			prob[m]*=0.5;
			if(maximum<prob[m])maximum=prob[m],c=m;
		}
		for(m=0,x=0;m<gmix->num;m++){ /* Do not use Viterbi aproximation. */
			tmpd=prob[m]-maximum;
			if(tmpd>mepsilon)x+=exp(tmpd); /* Use machine epsilon to avoid make exp's. */
		}
		llh+=(rmean=maximum+log(x)); tfreq[c]++;
		for(m=0;m<gmix->num;m++){ /* Accumulate counts of the sample in memory. */
			tmpd=prob[m]-rmean;
			if(tmpd>mepsilon){ /* Use machine epsilon to avoid this step. */
				zval[m]+=(tz=exp(tmpd)); inc=m*j;
				for(j=0;j<gmix->dimension;j++){
					mean[inc+j]+=(x=tz*feas->data[i][j]);
					dcov[inc+j]+=x*feas->data[i][j];
				}
			}
		}
	}
	pthread_mutex_lock(t->mutex); /* Accumulate counts obtained to the mixture. */
	gmix->llh+=llh;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m]._z+=zval[m]; inc=m*j;
		gmix->mix[m].freq+=tfreq[m];
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m]._mean[j]+=mean[inc+j];
			gmix->mix[m]._dcov[j]+=dcov[inc+j];
		}
	}
	pthread_mutex_unlock(t->mutex);
	free(zval); free(mean);
}

/* Perform one iteration of the EM algorithm with the data and the mixture indicated. */
decimal gmm_EMtrain(data *feas,gmm *gmix,number numthreads){
	pthread_mutex_t *mutex=(pthread_mutex_t*)calloc(1,sizeof(pthread_mutex_t));
	trainer *t=(trainer*)calloc(numthreads,sizeof(trainer));
	number m,i,inc; decimal tz,x;
	/* Calculate expected value and accumulate the counts (E Step). */
	gmm_init_classifier(gmix);
	for(m=0;m<gmix->num;m++)gmix->mix[m].freq=0;
	pthread_mutex_init(mutex,NULL);
	inc=feas->samples/numthreads;
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel training. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].mutex=mutex,t[i].ini=i*inc;
		t[i].end=(i==numthreads-1)?(feas->samples):((i+1)*inc);
		pthread_create(&t[i].thread,NULL,thread_trainer,(void*)&t[i]);
	}
	for(i=0;i<numthreads;i++) /* Wait to the end of the parallel training. */
		pthread_join(t[i].thread,NULL);
	pthread_mutex_destroy(mutex);
	/* Estimate the new parameters of the Gaussian Mixture (M Step). */
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].prior=log((tz=gmix->mix[m]._z)/feas->samples);
		for(i=0;i<gmix->dimension;i++){
			gmix->mix[m].mean[i]=(x=gmix->mix[m]._mean[i]/tz);
			gmix->mix[m].dcov[i]=(gmix->mix[m]._dcov[i]/tz)-(x*x);
			if(gmix->mix[m].dcov[i]<gmix->mcov[i]) /* Smoothing covariances. */
				gmix->mix[m].dcov[i]=gmix->mcov[i];
		}
	}
	return gmix->llh/feas->samples;
}
