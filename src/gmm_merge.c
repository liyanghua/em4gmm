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
	pthread_mutex_t *mutex; /* Variable needed to do the barrier.  */
	pthread_cond_t *cond;   /* Variable needed to do the barrier.  */
	number *count;          /* Variable needed to do the barrier.  */
	data *feas;             /* Shared pointer to loaded samples.   */
	gmm *gmix;              /* Shared pointer to gaussian mixture. */
	number ini, num;        /* Initial and total part processed.   */
	decimal *norm;          /* Pointer to shared normalization.    */
	mergelist *mlst;        /* Shared pointer to the merge list.   */
	decimal u;              /* The merge threshold used to merge.  */
}merger;

/* Merge the components of one mixture based on the provided list. */
gmm *gmm_merge(gmm *gmix,mergelist *mlst){
	gmm *cloned=gmm_create(mlst->endmix,gmix->dimension);
	number m,j,p=0,y;
	for(m=0;m<gmix->num;m++){
		if(mlst->merge[m]>0){ /* If it is marked as merged, merge with the other. */
			y=mlst->merge[m];
			cloned->mix[m-p].prior=gmix->mix[m].prior+gmix->mix[y].prior;
			cloned->mix[m-p].freq=gmix->mix[m].freq+gmix->mix[y].freq;
			for(j=0;j<gmix->dimension;j++){
				cloned->mix[m-p].mean[j]=(gmix->mix[m].mean[j]+gmix->mix[y].mean[j])*0.5;
				cloned->mix[m-p].dcov[j]=(gmix->mix[m].dcov[j]+gmix->mix[y].dcov[j])*0.5;
			}
		}else if(mlst->merge[m]==-1)p++; /* If it is marked as merged, skip it.  */
		else if(mlst->merge[m]==0){ /* If it is not marked, copy on new mixture. */
			cloned->mix[m-p].prior=gmix->mix[m].prior;
			cloned->mix[m-p].freq=gmix->mix[m].freq;
			for(j=0;j<gmix->dimension;j++){
				cloned->mix[m-p].mean[j]=gmix->mix[m].mean[j];
				cloned->mix[m-p].dcov[j]=gmix->mix[m].dcov[j];
			}
		}
	}
	for(j=0;j<gmix->dimension;j++) /* Leave the minimum covariance as it. */
		cloned->mcov[j]=gmix->mcov[j];
	gmm_delete(gmix);
	return cloned;
}

/* Parallel implementation of the similarity algorithm. */
void *thread_merger(void *tdata){
	merger *t=(merger*)tdata;
	decimal *norb=(decimal*)calloc(t->feas->samples,sizeof(decimal));
	decimal x,prob,nmax,mdiv; number m,n,i,j;
	for(m=t->ini;m<t->gmix->num;m+=t->num){ /* Precalculate the normalization part once. */
		t->norm[m]=-HUGE_VAL;
		for(i=0;i<t->feas->samples;i++){
			prob=t->gmix->mix[m].cgauss;
			for(j=0;j<t->gmix->dimension;j++){
				x=t->feas->data[i][j]-t->gmix->mix[m].mean[j];
				prob-=(x*x)*t->gmix->mix[m].dcov[j];
			}
			prob=(prob+prob)*0.5;
			t->norm[m]=t->norm[m]>prob?t->norm[m]:prob;
		}
	}
	pthread_mutex_lock(t->mutex); /* Barrier to wait all the other threads. */
	if(++(*t->count)==t->num)pthread_cond_broadcast(t->cond);
	else pthread_cond_wait(t->cond,t->mutex);
	pthread_mutex_unlock(t->mutex);
	for(m=t->ini;m<(t->gmix->num-1);m+=t->num){ /* Fast calculate of the similarity. */
		for(i=0;i<t->feas->samples;i++){
			norb[i]=t->gmix->mix[m].cgauss;
			for(j=0;j<t->gmix->dimension;j++){
				x=t->feas->data[i][j]-t->gmix->mix[m].mean[j];
				norb[i]-=(x*x)*t->gmix->mix[m].dcov[j];
			}
		}
		for(n=m+1;n<t->gmix->num;n++){ /* Similarity only with the next components. */
			nmax=-HUGE_VAL,mdiv=t->norm[m]+t->norm[n];
			for(i=0;i<t->feas->samples;i++){
				if(norb[i]-mdiv<t->u)continue; /* Continue if we not pass the threshold. */
				prob=t->gmix->mix[n].cgauss;
				for(j=0;j<t->gmix->dimension;j++){
					x=t->feas->data[i][j]-t->gmix->mix[n].mean[j];
					prob-=(x*x)*t->gmix->mix[n].dcov[j];
				}
				prob=(norb[i]+prob)*0.5;
				nmax=nmax>prob?nmax:prob;
			}
			prob=((nmax+nmax)-mdiv)*0.5; /* Similarity between n and m. */
			if(prob>t->u){
				t->mlst->merge[m]=n,t->mlst->value[m]=prob;
				break;
			}
		}
	}
	free(norb);
}

/* Obtain a merge list based on the similarity of two components. */
mergelist *gmm_merge_list(data *feas,gmm *gmix,decimal u,number numthreads){
	pthread_mutex_t *mutex=(pthread_mutex_t*)calloc(1,sizeof(pthread_mutex_t));
	pthread_cond_t *cond=(pthread_cond_t*)calloc(1,sizeof(pthread_cond_t));
	mergelist *mlst=(mergelist*)calloc(1,sizeof(mergelist));
	mlst->merge=(number*)calloc(mlst->inimix=gmix->num,sizeof(number));
	mlst->value=(decimal*)calloc(mlst->endmix=gmix->num,sizeof(decimal));
	decimal *norm=(decimal*)calloc(gmix->num,sizeof(decimal));
	number *count=(number*)calloc(1,sizeof(number)),m,n,i;
	merger *t=(merger*)calloc(numthreads,sizeof(merger));
	gmm_init_classifier(gmix); u=log(u);
	pthread_mutex_init(mutex,NULL);
	pthread_cond_init(cond,NULL);
	for(i=0;i<numthreads;i++){ /* Set and launch the parallel computing. */
		t[i].feas=feas,t[i].gmix=gmix,t[i].norm=norm,t[i].ini=i;
		t[i].u=u,t[i].num=numthreads,t[i].mlst=mlst;
		t[i].cond=cond,t[i].mutex=mutex,t[i].count=count;
		pthread_create(&t[i].thread,NULL,thread_merger,(void*)&t[i]);
	}
	for(i=0;i<numthreads;i++) /* Wait to the end of parallel computing. */
		pthread_join(t[i].thread,NULL);
	for(m=0;m<gmix->num;m++){ /* Compose and normalize the merge list.  */
		n=mlst->merge[m];
		if(n>0){
			if(mlst->merge[n]==-1)mlst->merge[m]=0;
			else mlst->merge[n]=-1,mlst->value[n]=0,mlst->endmix--;
		}
		if(mlst->merge[m]==0) /* Delete unused components of the mixture. */
			if(gmix->mix[m].freq<=1)
				mlst->merge[m]=-1,mlst->value[m]=0,mlst->endmix--;
	}
	gmm_init_classifier(gmix);
	pthread_mutex_destroy(mutex);
	pthread_cond_destroy(cond);
	free(norm); free(count);
	return mlst;
}

/* Free the allocated memory of the merge list. */
void gmm_merge_delete(mergelist *mlst){
	free(mlst->merge);
	free(mlst->value);
	free(mlst);
}
