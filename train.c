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

#include "gmm.h"

int main(int argc,char *argv[]) {
	if(argc==4||argc==5){
		decimal last=INT_MIN,llh,sigma=0.1;
		number i;
		if(argc==5)sigma=atof(argv[4]);
		data *feas=feas_load(argv[2]);
		gmm *gmix=gmm_initialize(feas,atoi(argv[1]));
		for(i=1;i<101;i++){
			llh=gmm_EMtrain(feas,gmix);
			printf("Iteration: %03i    Improvement: %3i%c    LogLikelihood: %.3f\n",
				i,abs(round(-100*(llh-last)/last)),'%',llh);
			if(last-llh>-sigma||isnan(last-llh)) break;
			last=llh;
		}
		feas_delete(feas);
		gmm_init_classifier(gmix); /* Inicializamos para ganar velocidad al reconocer. */
		gmm_save(argv[3],gmix);
		gmm_delete(gmix);
	}else fprintf(stderr,"Usage: %s <mixtures> <features> <model> [sigma]\n",argv[0]);
	return 0;
}
