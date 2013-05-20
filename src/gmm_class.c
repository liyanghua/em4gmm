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
	decimal result;         /* Variable to store the result found. */
	data *feas;             /* Shared pointer to loaded samples.   */
	gmm *gmix;              /* Shared pointer to gaussian mixture. */
	gmm *gworld;            /* Shared pointer to gaussian mixture. */
	number ini, end;        /* Initial and final sample processed. */
	FILE *f;               /* The jSON gzip file to save the log. */
	number *flag;           /* A flag for the first line on file.  */
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
	number i,inc=feas->samples/numthreads; decimal result=0;
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel classify. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].gworld=gworld,t[i].ini=i*inc,t[i].mutex=NULL;
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
	decimal *mprob=(decimal*)calloc(t->gmix->num,sizeof(decimal));
	char *buffer=(char*)calloc(300+20*t->gmix->num,sizeof(char));
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
			mprob[m]=(prob-max2)*0.5;
		}
		t->result+=mprob[c];
		sprintf(buffer,"\n\t\t{ \"sample\": %i, \"class\": %i, ",i,c);
		sprintf(buffer,"%s\"lprob\": [ %.10f",buffer,mprob[0]);
		for(m=1;m<t->gmix->num;m++)
			sprintf(buffer,"%s, %.10f",buffer,mprob[m]);
		sprintf(buffer,"%s ] }",buffer);
		pthread_mutex_lock(t->mutex); /* Write the classifier log on the jSON file. */
		if(t->flag[0]==0){
			fprintf(t->f,"%s",buffer);
			t->flag[0]=1;
		}else fprintf(t->f,",%s",buffer);
		pthread_mutex_unlock(t->mutex);
	}
	free(mprob);
	free(buffer);
}

/* Detailed Gaussian Mixture classifier using a Viterbi aproximation. */
decimal gmm_classify(char *filename,data *feas,gmm *gmix,gmm *gworld,number numthreads){
	pthread_mutex_t *mutex=(pthread_mutex_t*)calloc(1,sizeof(pthread_mutex_t));
	classifier *t=(classifier*)calloc(numthreads,sizeof(classifier));
	number i,inc=feas->samples/numthreads; decimal result=0;
	number *flag=(number*)calloc(1,sizeof(number));
	pthread_mutex_init(mutex,NULL);
	FILE *f=fopen(filename,"w");
	if(!f)fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
	fprintf(f,"{\n\t\"samples\": %i,\n\t\"classes\": %i,",feas->samples,gmix->num);
	fprintf(f,"\n\t\"samples_results\": [ ");
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel classify. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].gworld=gworld,t[i].ini=i*inc,t[i].mutex=mutex;
		t[i].end=(i==numthreads-1)?(feas->samples):((i+1)*inc),t[i].f=f,t[i].flag=flag;
		pthread_create(&t[i].thread,NULL,thread_classifier,(void*)&t[i]);
	}
	for(i=0;i<numthreads;i++){ /* Wait to the end of the parallel classify. */
		pthread_join(t[i].thread,NULL);
		result+=t[i].result;
	}
	fprintf(f,"\n\t],\n\t\"global_score\": %.10f",result);
	fprintf(f,"\n}"); fclose(f);
	return result/feas->samples;
}
