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
	pthread_t thread; /* pthread identifier of the current thread. */
	char *file;	/* Complete path of the file with the model used.  */
	data *feas; /* Shared pointer (read-only) to loaded samples.   */
	decimal result; /* Decimal used to store the thread's result.  */
}threadinfo;

/* Do a classification with the data and the model on parallel thread. */
void *thread(void *tdata){
	threadinfo *info=(threadinfo*)tdata; /* Get the data for the thread. */
	gmm *gmix=gmm_load(info->file); /* Load the model form the file.    */
	info->result=gmm_classify(info->feas,gmix); /* Compute probability. */
	gmm_delete(gmix);
	pthread_exit(NULL);
}

int main(int argc,char *argv[]){
	if(argc==3|argc==4){
		threadinfo t1,t2;
		data *feas=feas_load(argv[1]); /* Load the data from the file.   */
		if(argc==4){ /* Only applies if there are a world model defined. */
			t2.file=argv[3]; t2.feas=feas;
			pthread_create(&t2.thread,NULL,thread,(void*)&t2);
		}
		t1.file=argv[2]; t1.feas=feas;
		pthread_create(&t1.thread,NULL,thread,(void*)&t1);
		pthread_join(t1.thread,NULL); /* We wait to the end of threads. */
		if(argc==4){
			pthread_join(t2.thread,NULL); /* Wait to the other thread.  */
			t1.result-=t2.result;
		}
		fprintf(stdout,"LogProbability: %.10f\n",t1.result);
		feas_delete(feas);
	}else fprintf(stderr,"Usage: %s <features> <model> [world]\n",argv[0]);
	return 0;
}
