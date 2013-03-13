/* Algoritmo EM eficiente para GMM por Juan Daniel Valor Miro bajo la GPLv2. */
#include "gmm.h"

/* Clasificador de mixturas eficiente que usa una aproximacion a la Viterbi. */
decimal gmm_classify(data *feas,gmm *gmix){
	decimal x,maximum,prob,s=0;
	number i,m,j;
	for(i=0;i<feas->samples;i++){
		maximum=-HUGE_VAL;
		for(m=0;m<gmix->num;m++){
			prob=gmix->mix[m].cgauss; /* La parte invariante se calcula previamente. */
			for(j=0;j<gmix->dimension;j++){
				if(maximum>prob)break; /* Poda para acelerar el clasificador. */
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob-=(x*x)*gmix->mix[m].dcov[j];
			}
			if(maximum<prob)maximum=prob;
		}
		s+=maximum;
	}
	return (s*0.5)/feas->samples;
}

/* Inicializamos el clasificador calculando la parte invariante a los datos. */
decimal gmm_init_classifier(gmm *gmix){
	decimal cache=gmix->dimension*(-0.5)*log(2*M_PI);
	number m,j;
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].cgauss=0;
		for(j=0;j<gmix->dimension;j++){
			gmix->mix[m].cgauss+=log(gmix->mix[m].dcov[j]);
			gmix->mix[m].dcov[j]=1/gmix->mix[m].dcov[j];
			gmix->mix[m]._mean[j]=gmix->mix[m]._dcov[j]=0; /* A 0 las cuentas del EM. */
		}
		gmix->mix[m].cgauss=2*(gmix->mix[m].prior-((gmix->mix[m].cgauss*0.5)+cache));
	}
}

/* Realiza una iteracion del algoritmo EM con los datos y la mixtura indicados. */
decimal gmm_EMtrain(data *feas,gmm *gmix){
	decimal *prob=(decimal*)calloc(gmix->num<<1,sizeof(decimal));
	decimal tz,mean,llh=0,*z=prob+gmix->num,x,maximum;
	number i,m,j;
	/* Calculamos el valor esperado y acumulamos las cuentas (Paso E). */
	gmm_init_classifier(gmix); /* El primer paso del EM es clasificar. */
	for(i=0;i<feas->samples;i++){
		maximum=-HUGE_VAL;
		for(m=0;m<gmix->num;m++){
			prob[m]=gmix->mix[m].cgauss;
			for(j=0;j<gmix->dimension;j++){
				x=feas->data[i][j]-gmix->mix[m].mean[j];
				prob[m]-=(x*x)*gmix->mix[m].dcov[j];
			}
			prob[m]*=0.5;
			maximum=maximum>prob[m]?maximum:prob[m];
		}
		for(m=0,x=0;m<gmix->num;m++) /* No clasificamos a la Viterbi. */
			x+=exp(prob[m]-maximum);
		llh+=(mean=maximum+log(x));
		for(m=0;m<gmix->num;m++){
			z[m]+=(tz=exp(prob[m]-mean));
			for(j=0;j<feas->dimension;j++){ /* Acumulamos las cuentas. */
				gmix->mix[m]._mean[j]+=(x=tz*feas->data[i][j]);
				gmix->mix[m]._dcov[j]+=x*feas->data[i][j];
			}
		}
	}
	/* Estimamos los nuevos parametros de la distribucion gausiana (Paso M). */
	for(m=0;m<gmix->num;m++){
		gmix->mix[m].prior=log(z[m]/feas->samples);
		for(j=0;j<feas->dimension;j++){
			gmix->mix[m].mean[j]=(x=gmix->mix[m]._mean[j]/z[m]);
			gmix->mix[m].dcov[j]=(gmix->mix[m]._dcov[j]/z[m])-(x*x);
			if(gmix->mix[m].dcov[j]<gmix->mcov[j]) /* Suavizado de covarianzas. */
				gmix->mix[m].dcov[j]=gmix->mcov[j];
		}
	}
	free(prob);
	return llh/feas->samples;
}

/* Creamos una mixtura reservando memoria de forma contigua. */
inline gmm *gmm_create(number n,number d){
	gmm *gmix=(gmm*)calloc(1,sizeof(gmm));
	gmix->dimension=d;
	gmix->mcov=(decimal*)calloc(gmix->dimension,sizeof(decimal));
	gmix->mix=(gauss*)calloc(gmix->num=n,sizeof(gauss));
	for(n=0;n<gmix->num;n++){
		gmix->mix[n].mean=(decimal*)calloc(gmix->dimension*4,sizeof(decimal));
		gmix->mix[n].dcov=gmix->mix[n].mean+gmix->dimension;
		gmix->mix[n]._mean=gmix->mix[n].dcov+gmix->dimension;
		gmix->mix[n]._dcov=gmix->mix[n]._mean+gmix->dimension;
	}
	return gmix;
}

/* Creamos e inicializamos la mixtura con máxima verosimilitud y perturbando medias. */
gmm *gmm_initialize(data *feas,number nmix){
	gmm *gmix=gmm_create(nmix,feas->dimension);
	decimal x=1.0/gmix->num,y;
	number i,j,b=feas->samples/gmix->num,bc=0,k;
	/* Inicializacion de la primera mixtura con maxima verosimilitud. */
	gmix->mix[0].prior=log(x);
	for(j=0;j<feas->dimension;j++){
		for(i=0;i<feas->samples;i++){
			gmix->mix[0].mean[j]+=feas->data[i][j];
			gmix->mix[0].dcov[j]+=feas->data[i][j]*feas->data[i][j];
		}
		gmix->mix[0].mean[j]/=feas->samples;
		gmix->mix[0].dcov[j]=x*((gmix->mix[0].dcov[j]/feas->samples)
			-(gmix->mix[0].mean[j]*gmix->mix[0].mean[j]));
		gmix->mcov[j]=0.001*gmix->mix[0].dcov[j];
	}
	/* Perturbamos todas las medias con las medias de C bloques de muestras. */	
	for(i=gmix->num-1;i>=0;i--,bc+=b){
		gmix->mix[i].prior=gmix->mix[0].prior;
		for(k=0;k<feas->dimension;k++){
			for(j=bc,x=0;j<bc+b;j++)
				x+=feas->data[j][k];
			gmix->mix[i].mean[k]=(gmix->mix[0].mean[k]*0.9)+(0.1*x/b);
			gmix->mix[i].dcov[k]=gmix->mix[0].dcov[k];
		}
	}
	return gmix;
}

/* Creamos y cargamos la mixtura del fichero recibido como parámetro. */
gmm *gmm_load(char *filename){
	number m,d;
	FILE *f=fopen(filename,"rb");
	if(!f) fprintf(stderr,"Error: Not %s model file found.\n",filename),exit(1);
	fread(&d,1,sizeof(number),f);
	fread(&m,1,sizeof(number),f);
	gmm *gmix=gmm_create(m,d);
	fread(gmix->mcov,sizeof(decimal)*gmix->dimension,1,f);
	for(m=0;m<gmix->num;m++){
		fread(&gmix->mix[m].prior,sizeof(decimal),1,f);
		fread(&gmix->mix[m].cgauss,sizeof(decimal),1,f);
		fread(gmix->mix[m].mean,gmix->dimension*sizeof(decimal)*2,1,f);
	}
	fclose(f);
	return gmix;
}

/* Guardamos toda la mixtura en el fichero que se nos indica. */
void gmm_save(char *filename,gmm *gmix){
	number m;
	FILE *f=fopen(filename,"wb");
	if(!f) fprintf(stderr,"Error: Can not write to %s file.\n",filename),exit(1);
	fwrite(&gmix->dimension,1,sizeof(number),f);
	fwrite(&gmix->num,1,sizeof(number),f);
	fwrite(gmix->mcov,sizeof(decimal)*gmix->dimension,1,f);
	for(m=0;m<gmix->num;m++){
		fwrite(&gmix->mix[m].prior,sizeof(decimal),1,f);
		fwrite(&gmix->mix[m].cgauss,sizeof(decimal),1,f);
		fwrite(gmix->mix[m].mean,sizeof(decimal)*gmix->dimension*2,1,f);
	}
	fclose(f);
}

/* Liberamos la memoria reservada al crear la mixtura. */
void gmm_delete(gmm *gmix){
	number i;
	for(i=0;i<gmix->num;i++)
		free(gmix->mix[i].mean);
	free(gmix->mix);
	free(gmix);
}

/* Creamos y cargamos las muestras de un fichero de texto con formato. */
data *feas_load(char *filename){
	data *feas=(data*)calloc(1,sizeof(data));
	number i,j;
	char header[9];
	FILE *f=fopen(filename,"r");
	if(!f) fprintf(stderr,"Error: Not %s feature file found.\n",filename),exit(1);
	fscanf(f,"%s",header);
	if(strcmp(header,"AKREALTF")!=0)
		fprintf(stderr,"Error: Wrong %s feature file format.\n",filename),exit(1);
	fscanf(f,"%i",&feas->dimension);
	fscanf(f,"%i",&feas->samples);
	feas->data=(decimal**)malloc(feas->samples*sizeof(decimal*));
	decimal *aux=(decimal*)malloc(feas->dimension*feas->samples*sizeof(decimal));
	for(i=0;i<feas->samples;i++){
		feas->data[i]=aux;
		aux+=feas->dimension;
		for(j=0;j<feas->dimension;j++)
			fscanf(f,"%lf",&feas->data[i][j]);
	}
	fclose(f);
	return feas;
}

/* Liberamos la memoria reservada al crear las muestras. */
void feas_delete(data *feas){
	free(feas->data[0]);
	free(feas->data);
	free(feas);
}
