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

/* Show the classifier help message. */
void show_help(char *filename){
	fprintf(stderr,"Usage: %s <options>\n",filename);
	fprintf(stderr,"  Required:\n");
	fprintf(stderr,"    -d file.txt|file.gz   file that contains the samples vectors\n");
	fprintf(stderr,"    -m file.gmm           file of the trained model used to classify\n");
	fprintf(stderr,"  Recommended:\n");
	fprintf(stderr,"    -w file.gmm           optional world model used to smooth\n");
	fprintf(stderr,"    -r file.json          optional file to save the classify log\n");
	fprintf(stderr,"  Optional:\n");
	fprintf(stderr,"    -t 1-128              optional maximum number of threads used\n");
	fprintf(stderr,"    -h                    optional argument that shows this message\n");
}

/* Main execution of the classifier. */
int main(int argc,char *argv[]){
	number i,o,x=0; char *fnr=NULL,*fnf=NULL,*fnm=NULL,*fnw=NULL;
	while((o=getopt(argc,argv,"d:m:w:r:h"))!=-1){
		switch(o){
			case 'r': fnr=optarg; break;
			case 'd': fnf=optarg,x++; break;
			case 'm': fnm=optarg,x++; break;
			case 'w': fnw=optarg; break;
			case 'h': show_help(argv[0]),exit(1); break;
		}
	}
	if(x<2)show_help(argv[0]),exit(1); /* Test if exists needed arguments. */
	data *feas=feas_load(fnf); /* Load the data from the specified file.   */
	gmm *gmix=NULL,*gworld=NULL;
	if(fnw!=NULL)gworld=gmm_load(fnw); /* Load world model if is defined.  */
	gmix=gmm_load(fnm);
	cluster *c=gmm_classify(feas,gmix,gworld,NUM_THREADS);
	gmm_delete(gmix);
	if(gworld!=NULL)gmm_delete(gworld);
	fprintf(stdout,"Score: %.10f\n",c->result);
	if(fnr!=NULL)gmm_results_save(fnr,c); /* Save jSON log if is defined.  */
	gmm_results_delete(c);
	feas_delete(feas);
	return 0;
}
