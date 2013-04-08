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
		decimal _z;     /* Counts to estimate the future parameter (used by EM). */
	}gauss;

	typedef struct{
		gauss *mix;       /* Vector of gaussian distributions (above struct).    */
		number dimension; /* Number of dimensions of the gaussian mixture model. */
		number num;       /* Number of components of the gaussian mixture model. */
		decimal *mcov;    /* Minimum allowed covariances to avoid singularities. */
		decimal llh;      /* LogLikelihood after the training with EM algorithm. */
	}gmm;

	typedef struct{
		number samples;  /* Number of samples on the overall data.   */
		number mixtures; /* Number of gaussian mixtures trained.     */
		number *mix;     /* The class assigned for each data sample. */
		number *freq;    /* The number of samples in each mixture.   */
		decimal *prob;   /* The maximum score for each data sample.  */
		decimal result;  /* Variable to store the result computed.   */
	}cluster;

	typedef struct{
		number inimix;  /* The initial number of mixture components.  */
		number endmix;  /* The final number of mixture components.    */
		number *merge;  /* An index vector for the merged components. */
		decimal *value; /* Similarity computed on the merge vector.   */
	}mergelist;

	/* Public functions prototypes to work with Gaussian Mixture Models. */
	void gmm_save(char*,gmm*);
	gmm *gmm_load(char*);
	gmm *gmm_create(number,number);
	gmm *gmm_initialize(data*,number);
	void gmm_delete(gmm*);
	void gmm_init_classifier(gmm*);
	cluster *gmm_classify(data*,gmm*,gmm*,number);
	decimal gmm_EMtrain(data*,gmm*,number);
	void gmm_results_save(char*,cluster*);
	void gmm_results_delete(cluster*);
	mergelist *gmm_merge_list(data*,gmm*,decimal,number);
	gmm *gmm_merge(gmm*,mergelist*);
	void gmm_merge_delete(mergelist*);

#endif
