/* Algoritmo EM eficiente para GMM por Juan Daniel Valor Miro bajo la GPLv2. */
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
