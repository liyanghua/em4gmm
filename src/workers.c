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
#include "workers.h"

void *workers_thread(void *tdata){
	workers *t=(workers*)tdata;
	void (*routine_exec)(void*),*data;
	pthread_mutex_lock(&t->excluder);
	while(1){
		if(t->next==1){
			routine_exec=t->routine,data=t->data,t->next=0;
			pthread_cond_signal(&t->launcher);
			pthread_mutex_unlock(&t->excluder);
			routine_exec(data);
			pthread_mutex_lock(&t->excluder);
			if((--t->exec)==0)
				pthread_cond_signal(&t->waiter);
		}else pthread_cond_wait(&t->newtask,&t->excluder);
		if(t->stop==1)break;
	}
	pthread_mutex_unlock(&t->excluder);
}

void workers_waitall(workers *pool){
	pthread_mutex_lock(&pool->excluder);
	if(pool->exec!=0)
		pthread_cond_wait(&pool->waiter,&pool->excluder);
	pthread_mutex_unlock(&pool->excluder);
}

void workers_addtask(workers *pool,void (*routine)(void*),void *data){
	pthread_mutex_lock(&pool->excluder);
	while(1){
		if(pool->next==0){
			pool->data=data,pool->routine=routine;
			pool->next=1,pool->exec++;
			pthread_cond_signal(&pool->newtask);
			break;
		}else pthread_cond_wait(&pool->launcher,&pool->excluder);
	}
	pthread_mutex_unlock(&pool->excluder);
}

workers *workers_create(number num){
	workers *pool=(workers*)calloc(1,sizeof(workers));
	pool->threads=(pthread_t*)calloc(pool->num=num,sizeof(pthread_t));
	pthread_mutex_init(&pool->excluder,NULL);
	pthread_cond_init(&pool->newtask,NULL);
	pthread_cond_init(&pool->launcher,NULL);
	pthread_cond_init(&pool->waiter,NULL);
	number i;
	for(i=0;i<num;i++)
		pthread_create(&pool->threads[i],NULL,workers_thread,(void*)pool);
	return pool;
}

void workers_finish(workers *pool){
	number i;
	pthread_mutex_lock(&pool->excluder);
	pool->stop=1;
	pthread_cond_broadcast(&pool->newtask);
	pthread_mutex_unlock(&pool->excluder);
	for(i=0;i<pool->num;i++)
		pthread_join(pool->threads[i],NULL);
	pthread_cond_destroy(&pool->newtask);
	pthread_cond_destroy(&pool->launcher);
	pthread_cond_destroy(&pool->waiter);
	pthread_mutex_destroy(&pool->excluder);
	free(pool->threads);
	free(pool);
}
