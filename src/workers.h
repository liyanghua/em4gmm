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

#ifndef _workers_h
#define _workers_h

	typedef struct{
		pthread_t *threads;
		pthread_mutex_t excluder;
		pthread_cond_t newtask;
		pthread_cond_t launcher;
		pthread_cond_t waiter;
		number num,stop,next,exec;
		void (*routine)(void*),*data;
	}workers;

	workers *workers_create(number);
	void workers_addtask(workers*,void(*)(void*),void*);
	void workers_waitall(workers*);
	void workers_finish(workers*);

#endif
