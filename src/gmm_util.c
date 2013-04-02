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

gmm *gmm_merge(gmm *gmix,mergelist *mlst){
	number m,j,p=0,y;
	gmm *cloned=gmm_create(mlst->endmix,gmix->dimension);
	for(m=0;m<gmix->num;m++){
		if(mlst->merge[m]>0){
			y=mlst->merge[m];
			cloned->mix[m-p].prior=gmix->mix[m].prior+gmix->mix[y].prior;
			for(j=0;j<gmix->dimension;j++){
				cloned->mix[m-p].mean[j]=(gmix->mix[m].mean[j]+gmix->mix[y].mean[j])*0.5;
				cloned->mix[m-p].dcov[j]=(gmix->mix[m].dcov[j]+gmix->mix[y].dcov[j])*0.5;
			}
		}else if(mlst->merge[m]==-1){
			p++;
		}else if(mlst->merge[m]==0){
			cloned->mix[m-p].prior=gmix->mix[m].prior;
			for(j=0;j<gmix->dimension;j++){
				cloned->mix[m-p].mean[j]=gmix->mix[m].mean[j];
				cloned->mix[m-p].dcov[j]=gmix->mix[m].dcov[j];
			}
		}
	}
	for(j=0;j<gmix->dimension;j++)
		cloned->mcov[j]=gmix->mcov[j];
	gmm_delete(gmix);
	return cloned;
}

mergelist *gmm_merge_list(data *feas,gmm *gmix,decimal u){
	decimal x,prob,nmax; number n,m,i,j,inc,inx;
	mergelist *mlst=(mergelist*)calloc(1,sizeof(mergelist));
	mlst->merge=(number*)calloc(mlst->inimix=gmix->num,sizeof(number));
	mlst->value=(decimal*)calloc(mlst->endmix=gmix->num,sizeof(decimal));
	decimal *crit=(decimal*)calloc(gmix->num*feas->samples,sizeof(decimal));
	decimal *norm=(decimal*)calloc(gmix->num,sizeof(decimal));
	decimal *nort=(decimal*)calloc(feas->samples,sizeof(decimal));
	for(m=0;m<gmix->num;m++){
		inc=m*feas->samples,nmax=-HUGE_VAL;
		for(i=0;i<feas->samples;i++){
			prob=gmix->mix[m].cgauss; /* The precalculated non-data dependant part. */
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob-=(x*x)/gmix->mix[m].dcov[j];
			}
			crit[inc+i]=prob*0.5;
			nort[i]=crit[inc+i]+crit[inc+i];
			nmax=nmax>nort[i]?nmax:nort[i];
		}
		for(i=0;i<feas->samples;i++)
			norm[m]+=exp(nort[i]-nmax);
		norm[m]=nmax+log(norm[m]);
	}
	for(m=0;m<gmix->num-1;m++){
		if(mlst->merge[m]==-1)continue; /* Skip components that will be merged. */
		for(n=m+1;n<gmix->num;n++){
			if(mlst->merge[n]==-1)continue;
			inc=m*feas->samples,inx=n*feas->samples,prob=0;
			for(i=0;i<feas->samples;i++){
				nort[i]=crit[inc+i]+crit[inx+i];
				nmax=nmax>nort[i]?nmax:nort[i];
			}
			for(i=0;i<feas->samples;i++)
				prob+=exp(nort[i]-nmax);
			prob=nmax+log(prob);
			prob=exp(((prob+prob)-(norm[m]+norm[n]))*0.5);
			if(prob>u){
				mlst->merge[m]=n;
				mlst->merge[n]=-1;
				mlst->value[m]=prob;
				mlst->endmix--;
				break;
			}
		}
	}
	free(crit);
	free(norm);
	free(nort);
	return mlst;
}
