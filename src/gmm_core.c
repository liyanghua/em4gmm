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
	pthread_t thread; /* pthread identifier of the thread.   */
	decimal result;   /* Variable to store the result found. */
	data *feas;       /* Shared pointer to loaded samples.   */
	gmm *gmix;        /* Shared pointer to gaussian mixture. */
	gmm *gworld;      /* Shared pointer to gaussian mixture. */
	number ini, end;  /* Initial and final sample processed. */
	cluster *c;       /* The cluster of the data classified. */
}classifier;

typedef struct{
	pthread_t thread;       /* pthread identifier of the thread.   */
	pthread_mutex_t *mutex; /* Common mutex to lock shared data.   */
	data *feas;             /* Shared pointer to loaded samples.   */
	gmm *gmix;              /* Shared pointer to gaussian mixture. */
	number ini, end;        /* Initial and final sample processed. */
}trainer;

/* Initialize the classifier by calculating the non-data dependant part. */
void gmm_init_classifier(gmm *gmix){
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

/* Parallel implementation of the Gaussian Mixture classifier. */
void *thread_classifier(void *tdata){
	classifier *t=(classifier*)tdata;
	decimal x,max1,max2,prob; number i,m,j,c;
	for(i=t->ini;i<t->end;i++){
		max1=-HUGE_VAL,c=-1;
		for(m=0;m<t->gmix->num;m++){
			prob=t->gmix->mix[m].cgauss; /* The precalculated non-data dependant part. */
			for(j=0;j<t->gmix->dimension;j++){
				x=t->feas->data[i][j]-t->gmix->mix[m].mean[j];
				prob-=(x*x)*t->gmix->mix[m].dcov[j];
			}
			if(max1<prob)max1=prob,c=m; /* Fast classifier using Viterbi aproximation. */
		}
		if(t->gworld!=NULL){ /* If the world model is defined, use it. */
			max2=-HUGE_VAL;
			for(m=0;m<t->gworld->num;m++){
				prob=t->gworld->mix[m].cgauss;
				for(j=0;j<t->gworld->dimension;j++){
					x=t->feas->data[i][j]-t->gworld->mix[m].mean[j];
					prob-=(x*x)*t->gworld->mix[m].dcov[j];
				}
				if(max2<prob)max2=prob;
			}
		}else max2=0;
		t->c->prob[i]=(max1-max2)*0.5,t->c->mix[i]=c,t->c->freq[c]+=1;
		t->result+=t->c->prob[i]; /* Compute final probability. */
	}
}

/* Efficient Gaussian Mixture classifier using a Viterbi aproximation. */
cluster *gmm_classify(data *feas,gmm *gmix,gmm *gworld,number numthreads){
	classifier *t=(classifier*)calloc(numthreads,sizeof(classifier));
	cluster *c=(cluster*)calloc(1,sizeof(cluster));
	c->mix=(number*)calloc(c->samples=feas->samples,sizeof(number));
	c->freq=(number*)calloc(c->mixtures=gmix->num,sizeof(number));
	c->prob=(decimal*)calloc(c->samples,sizeof(decimal)),c->result=0;
	number i,inc=feas->samples/numthreads;
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel classify. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].gworld=gworld,t[i].ini=i*inc,t[i].c=c;
		t[i].end=(i==numthreads-1)?(feas->samples):((i+1)*inc);
		pthread_create(&t[i].thread,NULL,thread_classifier,(void*)&t[i]);
	}
	for(i=0;i<numthreads;i++){ /* Wait to the end of the parallel classify. */
		pthread_join(t[i].thread,NULL);
		c->result+=t[i].result;
	}
	c->result/=feas->samples;
	return c;
}

/* Parallel implementation of the E Step of the EM algorithm. */
void *thread_trainer(void *tdata){
	trainer *t=(trainer*)tdata; /* Get the data for the thread and alloc memory. */
	gmm *gmix=t->gmix; data *feas=t->feas;
	decimal *zval=(decimal*)calloc(2*gmix->num,sizeof(decimal)),*prob=zval+gmix->num;
	decimal *mean=(decimal*)calloc(2*gmix->num*gmix->dimension,sizeof(decimal));
	decimal *dcov=mean+(gmix->num*gmix->dimension),llh=0,x,tz,rmean,maximum;
	number i,j,m,inc;
	for(i=t->ini;i<t->end;i++){
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
	pthread_mutex_lock(t->mutex); /* Accumulate counts obtained to the mixture. */
	gmix->llh+=llh;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m]._z+=zval[m]; inc=m*j;
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
