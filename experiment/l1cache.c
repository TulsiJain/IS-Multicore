#include<stdio.h> 
#include<string.h> 
#include<pthread.h> 
#include<stdlib.h> 
#include<unistd.h> 

pthread_mutex_t lock; 
int list_size = 100000000;
int thread_count;
int array[100000000];


void* multiThread_Handler(void *arg) 
{
	int thread_index = *((int *)arg);
	pthread_mutex_lock(&lock); 
	unsigned int start_index = (thread_index*(list_size/thread_count));
	unsigned int end_index = ((thread_index+1)*(list_size/thread_count));
	pthread_mutex_unlock(&lock); 

	for (int i =  start_index; i < end_index - 12;  i++) {
		int a = array[i+1] + array[i+2] + array[i+3] + array[i+4]
		    + array[i+5] + array[i+6] + array[i+7] + array[i+8]
		    + array[i+9] + array[i+10] + array[i+11] + array[i+12];
		array[i] = array[i] + a%100;
		array[i] = array[i]%100;
	}
	return NULL;
} 

int main(int argc, char *argv[]) 
{
	thread_count = atoi(argv[1]); 
	pthread_t threads[thread_count];
	for (int i = 0; i < thread_count; i++) {
		int *arg = (int *) malloc(sizeof(*arg));
		*arg = i;
		int error = pthread_create(&(threads[i]), NULL,
		    multiThread_Handler, arg);
		if (error != 0)
			printf("\nCan't create thread :[%s]", strerror(error));
	}

	for (int i = 0; i< thread_count; i++) {
		pthread_join(threads[i], NULL);
	}
			exit(0); 
} 
