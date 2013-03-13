/* Algoritmo EM eficiente para GMM por Juan Daniel Valor Miro bajo la GPLv2. */
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
