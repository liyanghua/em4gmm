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

#ifndef _gmm_h
#define _gmm_h

	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <limits.h>

	#ifndef M_PI
		#define M_PI 3.14159265358979323846
	#endif

	typedef double decimal;
	typedef int number;

	typedef struct{
		decimal **data;
		number samples;
		number dimension;
	}data;

	typedef struct{
		decimal prior;
		decimal cgauss; /* Cache para acelerar el clasificador. */
		decimal *mean;
		decimal *dcov; /* Para el reconocimiento son las inversas. */
		decimal *_mean; /* Cuentas del parametro futuro (usado por el EM). */
		decimal *_dcov; /* Cuentas del parametro futuro (usado por el EM). */
	}gauss;

	typedef struct{
		gauss *mix;
		number dimension;
		number num;
		decimal *mcov; /* Covarianzas minimas para evitar singularidades. */
	}gmm;

	void gmm_save(char*,gmm*);
	gmm *gmm_load(char*);
	gmm *gmm_initialize(data*,number);
	void gmm_delete(gmm*);
	decimal gmm_init_classifier(gmm*);
	decimal gmm_classify(data*,gmm*);
	decimal gmm_EMtrain(data*,gmm*);
	data *feas_load(char*);
	void feas_delete(data*);

#endif
