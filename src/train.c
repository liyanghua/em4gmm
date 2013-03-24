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

/* Show the trainer help message. */
void show_help(char *filename){
	fprintf(stderr,"Usage: %s <options>\n",filename);
	fprintf(stderr,"  -d file.txt|file.gz   file that contains all the samples vectors\n");
	fprintf(stderr,"  -m file.gmm           file used to save the trained mixture model\n");
	fprintf(stderr,"  -n 2-1048576          optional number of components of the mixture\n");
	fprintf(stderr,"  -s 0.0001-1.0         optional stop criterion based on likelihood\n");
	fprintf(stderr,"  -h                    optional argument that shows this message\n");
}

/* Main execution of the trainer. */
int main(int argc,char *argv[]) {
	number i,o,x=0,nmix=-1; char *fnf=NULL,*fnm=NULL;
	decimal last=INT_MIN,llh,sigma=0.1;
	while((o=getopt(argc,argv,"d:m:n:s:h"))!=-1){
		switch(o){
			case 'n': nmix=atoi(optarg); break;
			case 'd': fnf=optarg,x++; break;
			case 'm': fnm=optarg,x++; break;
			case 's': sigma=atof(optarg); break;
			case 'h': show_help(argv[0]),exit(1); break;
		}
	}
	if(x<2)show_help(argv[0]),exit(1); /* Test if exists all the needed arguments. */
	data *feas=feas_load(fnf); /* Load the features from the specified disc file.  */
	nmix=(nmix==-1)?sqrt(feas->samples/2):nmix;
	gmm *gmix=gmm_initialize(feas,nmix); /* Good GMM initialization using data.    */
	for(i=1;i<101;i++){ /* We set 100 maximum iterations to obtain EM convergence. */
		llh=gmm_EMtrain(feas,gmix,NUM_THREADS); /* Compute one iteration of EM.    */
		printf("Iteration: %03i    Improvement: %3i%c    LogLikelihood: %.3f\n",
			i,abs(round(-100*(llh-last)/last)),'%',llh); /* Show the EM results.   */
		if(last-llh>-sigma||isnan(last-llh)) break; /* Break with sigma threshold. */
		last=llh;
	}
	feas_delete(feas);
	gmm_init_classifier(gmix); /* Pre-compute the non-data dependant part of classifier. */
	gmm_save(fnm,gmix); /* Save the model with the pre-computed part for fast classify.  */
	gmm_delete(gmix);
	return 0;
}
