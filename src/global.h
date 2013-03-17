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

#ifndef _global_h
#define _global_h

	#include <stdio.h>
	#include <stdlib.h>
	#include <math.h>
	#include <limits.h>
	#include <pthread.h>

	#ifndef M_PI
		#define M_PI 3.14159265358979323846
	#endif

	typedef double decimal; /* Specifies the default decimal type. */
	typedef int number;     /* Specifies the default integer type. */

#endif