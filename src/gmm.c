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
		t[i].feas=feas; t[i].gmix=gmix; t[i].mutex=mutex; t[i].ini=i*inc;
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
	number i; FILE *f=fopen(filename,"w");
	if(!f)fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
	fprintf(f,"{\n\t\"global_score\": %.10f,",c->result);
	fprintf(f,"\n\t\"samples\": %i,\n\t\"mixtures\": %i,",c->samples,c->mixtures);
	fprintf(f,"\n\t\"mixture_occupation\": [ %i",c->freq[0]);
	for(i=1;i<c->mixtures;i++)
		fprintf(f,", %i",c->freq[i]);
	fprintf(f," ],\n\t\"samples_classification\": [ %i",c->mix[0]);
	for(i=1;i<c->samples;i++)
		fprintf(f,", %i",c->mix[i]);
	fprintf(f," ],\n\t\"samples_score\": [ %.10f",c->prob[0]);
	for(i=1;i<c->samples;i++)
		fprintf(f,", %.10f",c->prob[i]);
	fprintf(f," ]\n}");
	fclose(f);
}

/* Free the allocated memory of the classifier results. */
void gmm_results_delete(cluster *c){
	free(c->mix);
	free(c->freq);
	free(c->prob);
	free(c);
}
