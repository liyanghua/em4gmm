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

/* Main execution of the classifier. */
int main(int argc,char *argv[]){
	if(argc==3|argc==4){
		data *feas=feas_load(argv[1]); /* Load the data from the file.   */
		gmm *gmix=NULL,*gworld=NULL;
		if(argc==4)gworld=gmm_load(argv[3]);
		gmix=gmm_load(argv[2]);
		cluster *c=gmm_classify(feas,gmix,gworld,NUM_THREADS);
		gmm_delete(gmix);
		if(gworld!=NULL)gmm_delete(gworld);
		fprintf(stdout,"Score: %.10f\n",c->result);
		feas_delete(feas);
	}else fprintf(stderr,"Usage: %s <features> <model> [world]\n",argv[0]);
	return 0;
}
