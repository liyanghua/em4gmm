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

/* Parallel and fast implementation of the Gaussian Mixture classifier. */
void *thread_simple_classifier(void *tdata){
	classifier *t=(classifier*)tdata;
	decimal x,max1,max2,prob; number i,m,j;
	for(i=t->ini;i<t->end;i++){
		max1=-HUGE_VAL;
		for(m=0;m<t->gmix->num;m++){
			prob=t->gmix->mix[m].cgauss; /* The precalculated non-data dependant part. */
			for(j=0;j<t->gmix->dimension;j++){
				x=t->feas->data[i][j]-t->gmix->mix[m].mean[j];
				prob-=(x*x)*t->gmix->mix[m].dcov[j];
			}
			if(max1<prob)max1=prob; /* Fast classifier using Viterbi aproximation. */
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
		t->result+=max1-max2; /* Compute final probability. */
	}
	t->result*=0.5;
}

/* Efficient Gaussian Mixture classifier using a Viterbi aproximation. */
decimal gmm_simple_classify(data *feas,gmm *gmix,gmm *gworld,number numthreads){
	classifier *t=(classifier*)calloc(numthreads,sizeof(classifier));
	number i,inc=feas->samples/numthreads; decimal result;
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel classify. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].gworld=gworld,t[i].ini=i*inc,t[i].c=NULL;
		t[i].end=(i==numthreads-1)?(feas->samples):((i+1)*inc);
		pthread_create(&t[i].thread,NULL,thread_simple_classifier,(void*)&t[i]);
	}
	for(i=0;i<numthreads;i++){ /* Wait to the end of the parallel classify. */
		pthread_join(t[i].thread,NULL);
		result+=t[i].result;
	}
	return result/feas->samples;
}

/* Parallel implementation of the Gaussian Mixture classifier that holds the data. */
void *thread_classifier(void *tdata){
	classifier *t=(classifier*)tdata;
	decimal x,max1,max2,prob; number i,m,j,c;
	for(i=t->ini;i<t->end;i++){
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
		max1=-HUGE_VAL,c=-1;
		for(m=0;m<t->gmix->num;m++){
			prob=t->gmix->mix[m].cgauss; /* The precalculated non-data dependant part. */
			for(j=0;j<t->gmix->dimension;j++){
				x=t->feas->data[i][j]-t->gmix->mix[m].mean[j];
				prob-=(x*x)*t->gmix->mix[m].dcov[j];
			}
			if(max1<prob)max1=prob,c=m; /* Fast classifier using Viterbi aproximation. */
			t->c->prob[i*t->gmix->num+m]=(prob-max2)*0.5;
		}
		t->result+=(max1-max2)*0.5,t->c->mix[i]=c,t->c->freq[c]+=1;
	}
}

/* Detailed Gaussian Mixture classifier using a Viterbi aproximation. */
cluster *gmm_classify(data *feas,gmm *gmix,gmm *gworld,number numthreads){
	classifier *t=(classifier*)calloc(numthreads,sizeof(classifier));
	cluster *c=(cluster*)calloc(1,sizeof(cluster));
	c->mix=(number*)calloc(c->samples=feas->samples,sizeof(number));
	c->freq=(number*)calloc(c->mixtures=gmix->num,sizeof(number));
	c->prob=(decimal*)calloc(c->samples*c->mixtures,sizeof(decimal)),c->result=0;
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
