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
	fprintf(stderr,"  Required:\n");
	fprintf(stderr,"    -d file.txt|file.gz   file that contains all the samples vectors\n");
	fprintf(stderr,"    -m file.gmm           file used to save the trained mixture model\n");
	fprintf(stderr,"  Recommended:\n");
	fprintf(stderr,"    -n 2-524228           optional number of components of the mixture\n");
	fprintf(stderr,"  Optional:\n");
	fprintf(stderr,"    -u 0.0-1.0            optional merge threshold based on similarity\n");
	fprintf(stderr,"    -s 0.0-1.0            optional stop criterion based on likelihood\n");
	fprintf(stderr,"    -i 1-10000            optional maximum number of EM iterations\n");
	fprintf(stderr,"    -t 1-256              optional maximum number of threads used\n");
	fprintf(stderr,"    -h                    optional argument that shows this message\n");
}

/* Show an error and leave the program. */
void show_error(const char *message){
	fprintf(stderr,"Error: %s.\n",message),exit(1);
}

/* Main execution of the trainer. */
int main(int argc,char *argv[]) {
	number i,o,x=0,nmix=-1,imax=100,t=16; char *fnf=NULL,*fnm=NULL;
	decimal last=INT_MIN,llh,sigma=0.1,m=-1.0;
	while((o=getopt(argc,argv,"u:t:i:d:m:n:s:h"))!=-1){
		switch(o){
			case 't': t=atoi(optarg);
				if(t>256||t<1)show_error("Number of threads must be on the 1-256 range");
				break;
			case 'n': nmix=atoi(optarg);
				if(nmix>524228||nmix<2)show_error("Number of components must be on the 2-32768 range");
				break;
			case 'i': imax=atoi(optarg);
				if(imax>10000||imax<1)show_error("Number of iterations must be on the 1-10000 range");
				break;
			case 'd': fnf=optarg,x++; break;
			case 'm': fnm=optarg,x++; break;
			case 'u': m=atof(optarg);
				if(m>1.0||m<0.0)show_error("Merge threshold must be on the 0.0-1.0 range");
				break;
			case 's': sigma=atof(optarg);
				if(sigma>1.0||imax<0.0)show_error("Sigma criterion must be on the 0.0-1.0 range");
				break;
			case 'h': show_help(argv[0]),exit(1); break;
		}
	}
	if(x<2)show_help(argv[0]),exit(1); /* Test if exists all the needed arguments. */
	data *feas=feas_load(fnf); /* Load the features from the specified disc file.  */
	nmix=(nmix==-1)?sqrt(feas->samples/2):nmix;
	gmm *gmix=gmm_initialize(feas,nmix); /* Good GMM initialization using data.    */
	for(o=1;o<=imax;o++){
		for(i=1;i<=imax;i++){
			llh=gmm_EMtrain(feas,gmix,t); /* Compute one iteration of EM algorithm.   */
			fprintf(stdout,"Iteration: %05i    Improvement: %3i%c    LogLikelihood: %.3f\n",
				i,abs(round(-100*(llh-last)/last)),'%',llh); /* Show the EM results.  */
			if(last-llh>-sigma||isnan(last-llh))break; /* Break with sigma threshold. */
			last=llh;
		}
		x=gmix->num;
		if(m>=0){
			mergelist *mlst=gmm_merge_list(feas,gmix,m);
			gmix=gmm_merge(gmix,mlst);
			gmm_merge_delete(mlst);
			fprintf(stdout,"Number of Components: %06i   Merged: %06i\n",gmix->num,x-gmix->num);
		}
		if(x==gmix->num)break;
		last=INT_MIN;
	}
	feas_delete(feas);
	gmm_init_classifier(gmix); /* Pre-compute the non-data dependant part of classifier. */
	gmm_save(fnm,gmix); /* Save the model with the pre-computed part for fast classify.  */
	gmm_delete(gmix);
	return 0;
}
