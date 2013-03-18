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

	typedef struct{
		decimal prior;  /* Prior probability of this component of the mixture.   */
		decimal cgauss; /* Cache storage in order to make faster the classifier. */
		decimal *mean;  /* Means vector of gaussian multivariate distribution.   */
		decimal *dcov;  /* Diagonal covariances, when classify are the inverses. */
		decimal *_mean; /* Counts to estimate the future parameter (used by EM). */
		decimal *_dcov; /* Counts to estimate the future parameter (used by EM). */
		decimal _z;
	}gauss;

	typedef struct{
		gauss *mix;       /* Vector of gaussian distributions (above struct).    */
		number dimension; /* Number of dimensions of the gaussian mixture model. */
		number num;       /* Number of components of the gaussian mixture model. */
		decimal *mcov;    /* Minimum allowed covariances to avoid singularities. */
		decimal llh;
	}gmm;

	typedef struct{
		pthread_t thread; /* pthread identifier of the current thread. */
		pthread_mutex_t *mutex;
		data *feas; /* Shared pointer (read-only) to loaded samples.   */
		gmm *gmix;
		number ini;
		number end;
	}trainer;

	/* Public functions prototypes to work with Gaussian Mixture Models. */
	void gmm_save(char*,gmm*);
	gmm *gmm_load(char*);
	gmm *gmm_initialize(data*,number);
	void gmm_delete(gmm*);
	decimal gmm_init_classifier(gmm*);
	decimal gmm_classify(data*,gmm*);
	decimal gmm_EMtrain(data*,gmm*);

#endif
