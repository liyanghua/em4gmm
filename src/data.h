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

#ifndef _data_h
#define _data_h

	typedef struct{
		decimal **data;   /* A matrix of samples*dimension elements.  */
		number samples;   /* Number of samples on the overall data.   */
		number dimension; /* Number of dimensions of the data loaded. */
	}data;

	/* Public functions prototypes to work with the samples. */
	data *feas_load(char*);
	void feas_delete(data*);

#endif
