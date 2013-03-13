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

int main(int argc,char *argv[]){
	if(argc==3|argc==4){
		data *feas=feas_load(argv[1]);
		gmm *gmix=gmm_load(argv[2]);
		decimal result=gmm_classify(feas,gmix);
		gmm_delete(gmix);
		if(argc==4){ /* Si hay un modelo del mundo obtenemos su probabilidad. */
			gmix=gmm_load(argv[3]);
			result-=gmm_classify(feas,gmix);
			gmm_delete(gmix);
			fprintf(stdout,"Score: %.10f\n",result);
		}else fprintf(stdout,"LogProbability: %.10f\n",result);
		feas_delete(feas);
	}else fprintf(stderr,"Usage: %s <features> <model> [world]\n",argv[0]);
	return 0;
}
