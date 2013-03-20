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

/* Asynchronous implementation of the buffer to feas converter. */
void *thread_loader(void *tdata){
	loader *info=(loader*)tdata; number i;
	data *feas=info->feas; char *buff=info->buff;
	for(i=0;i<info->r;i++){
		if(buff[i]>='0'){ /* Convert an array into a decimal number. */
			info->next=(info->point==0)?((10*info->next)+buff[i]-'0'):
				(info->next+((info->dec*=0.1)*(buff[i]-'0'))),info->c++;
		}else if(buff[i]<'-'&&info->c>0){
			if(info->header==0){
				if(info->d==0){ /* If we start a new sample, set the memory. */
					feas->data[info->s]=info->aux;
					info->aux+=feas->dimension;
				}
				feas->data[info->s][info->d]=info->next*info->sign;
				feas->mean[info->d]+=feas->data[info->s][info->d];
				feas->variance[info->d]+=feas->data[info->s][info->d]*feas->data[info->s][info->d];
				if((++info->d)==feas->dimension)info->d=0,info->s++;
			}else if(info->header==1){ /* Finish header reading and alloc needed memory. */
				feas->samples=(int)info->next;
				feas->mean=(decimal*)calloc(2*feas->dimension,sizeof(decimal*));
				feas->variance=feas->mean+feas->dimension;
				feas->data=(decimal**)calloc(feas->samples,sizeof(decimal*));
				info->aux=(decimal*)calloc(feas->dimension*feas->samples,sizeof(decimal));
				info->header=info->s=info->d=0;
			}else if(info->header==2)feas->dimension=(int)info->next,info->header=1;
			info->sign=info->dec=1,info->next=info->point=info->c=0;
		}else if(buff[i]=='-')info->sign=-1;
		else if(buff[i]=='.')info->point=1;
	}
	pthread_exit(NULL);
}

/* Load the samples from a plain text file with the specified format. */
data *feas_load(char *filename){
	data *feas=(data*)calloc(1,sizeof(data));
	loader *t=(loader*)calloc(1,sizeof(loader));
	t->s=t->d=t->c=t->point=t->next=0,t->header=2,t->dec=t->sign=1;
	number i,r; t->feas=feas,t->buff=NULL;
	gzFile f=gzopen(filename,"r"); /* Read the file using zlib library. */
	gzbuffer(f,128*1024);
	if(!f) fprintf(stderr,"Error: Not %s feature file found.\n",filename),exit(1);
	while(!gzeof(f)){
		char *buff=(char*)calloc(SIZE_BUFFER,sizeof(char));
		r=gzread(f,buff,SIZE_BUFFER); /* Read the buffer and do asynchronous load. */
		pthread_join(t->thread,NULL);
		if(t->buff!=NULL)free(t->buff);
		t->buff=buff,t->r=r;
		pthread_create(&t->thread,NULL,thread_loader,(void*)t);
	}
	pthread_join(t->thread,NULL);
	free(t->buff); free(t);
	gzclose(f);
	for(i=0;i<feas->dimension;i++){ /* Compute the mean and variance of the data. */
		feas->mean[i]/=feas->samples;
		feas->variance[i]=(feas->variance[i]/feas->samples)-(feas->mean[i]*feas->mean[i]);
	}
	return feas;
}

/* Free the allocated memory of the samples. */
void feas_delete(data *feas){
	free(feas->data[0]);
	free(feas->data);
	free(feas->mean);
	free(feas);
}
