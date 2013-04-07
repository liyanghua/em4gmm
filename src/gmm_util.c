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

/* Merge the components of one mixture based on the provided list. */
gmm *gmm_merge(gmm *gmix,mergelist *mlst){
	number m,j,p=0,y;
	gmm *cloned=gmm_create(mlst->endmix,gmix->dimension);
	for(m=0;m<gmix->num;m++){
		if(mlst->merge[m]>0){ /* If it is marked as merged, merge with the other. */
			y=mlst->merge[m];
			cloned->mix[m-p].prior=gmix->mix[m].prior+gmix->mix[y].prior;
			for(j=0;j<gmix->dimension;j++){
				cloned->mix[m-p].mean[j]=(gmix->mix[m].mean[j]+gmix->mix[y].mean[j])*0.5;
				cloned->mix[m-p].dcov[j]=(gmix->mix[m].dcov[j]+gmix->mix[y].dcov[j])*0.5;
			}
		}else if(mlst->merge[m]==-1)p++; /* If it is marked as merged, skip it.  */
		else if(mlst->merge[m]==0){ /* If it is not marked, copy on new mixture. */
			cloned->mix[m-p].prior=gmix->mix[m].prior;
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

/* Obtain a merge list based on the similarity of two components. */
mergelist *gmm_merge_list(data *feas,gmm *gmix,decimal u,number numthreads){
	mergelist *mlst=(mergelist*)calloc(1,sizeof(mergelist));
	mlst->merge=(number*)calloc(mlst->inimix=gmix->num,sizeof(number));
	mlst->value=(decimal*)calloc(mlst->endmix=gmix->num,sizeof(decimal));
	decimal *norm=(decimal*)calloc(gmix->num,sizeof(decimal));
	decimal *norb=(decimal*)calloc(feas->samples,sizeof(decimal));
	decimal x,prob,nmax; number m,n,i,j;
	gmm_init_classifier(gmix); u=log(u);
	for(m=0;m<gmix->num;m++){ /* Precalculate the normalization part once. */
		norm[m]=-HUGE_VAL;
		for(i=0;i<feas->samples;i++){
			prob=gmix->mix[m].cgauss;
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob-=(x*x)*gmix->mix[m].dcov[j];
			}
			prob=(prob+prob)*0.5;
			norm[m]=norm[m]>prob?norm[m]:prob;
		}
	}
	for(m=0;m<gmix->num-1;m++){
		if(mlst->merge[m]==-1)continue; /* Skip components that will be merged. */
		for(i=0;i<feas->samples;i++){
			norb[i]=gmix->mix[m].cgauss;
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				norb[i]-=(x*x)*gmix->mix[m].dcov[j];
			}
		}
		for(n=m+1;n<gmix->num;n++){
			if(mlst->merge[n]==-1)continue; /* Skip components that will be merged. */
			nmax=-HUGE_VAL;
			for(i=0;i<feas->samples;i++){
				prob=gmix->mix[n].cgauss;
				for(j=0;j<gmix->dimension;j++){
					x=feas->data[i][j]-gmix->mix[n].mean[j];
					prob-=(x*x)*gmix->mix[n].dcov[j];
				}
				prob=(norb[i]+prob)*0.5;
				nmax=nmax>prob?nmax:prob;
			}
			prob=((nmax+nmax)-(norm[m]+norm[n]))*0.5; /* Similarity between n and m. */
			if(prob>u){
				mlst->merge[m]=n,mlst->merge[n]=-1;
				mlst->value[m]=prob,mlst->endmix--;
				break;
			}
		}
	}
	gmm_init_classifier(gmix);
	free(norm); free(norb);
	return mlst;
}

/* Free the allocated memory of the merge list. */
void gmm_merge_delete(mergelist *mlst){
	free(mlst->merge);
	free(mlst->value);
	free(mlst);
}
