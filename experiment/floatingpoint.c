#include<stdio.h> 
#include<string.h> 
#include<pthread.h> 
#include<stdlib.h> 
#include<unistd.h> 

pthread_mutex_t lock; 
int list_size = 100000000;
int thread_count = 8;
int array[100000000];


void* multiThread_Handler(void *arg) 
{ 
	// pthread_mutex_lock(&lock); 

	// unsigned long i = 0; 
	// counter += 1; 
	// printf("\n Job %d has started\n", counter); 

	// for(i=0; i<(0xFFFFFFFF);i++); 

	// printf("\n Job %d has finished\n", counter); 

	// pthread_mutex_unlock(&lock); 

	// return NULL;

	int thread_index = *((int *)arg);
   	printf("thread index %d\n", thread_index);

   	pthread_mutex_lock(&lock); 
   	unsigned int start_index = (thread_index*(list_size/thread_count));
   	unsigned int end_index = ((thread_index+1)*(list_size/thread_count));
   	pthread_mutex_unlock(&lock); 

   // unsigned int start_index = (thread_index*(list_size/thread_count));
   // unsigned int end_index = ((thread_index+1)*(list_size/thread_count));


   // std::cout << "Start Index: " << start_index << std::endl;
   // std::cout << "End Index: " << end_index << std::endl;
   // std::cout << "i: " << thread_index << std::endl;

   	for(int i =  start_index; i < end_index;  i++)
   	{
   		array[i] = array[i] + 1;
   		// printf("Processing array element at %d\n", i);
      // std::cout <<"Processing array element at: " << i << std::endl;
   	}
   	return NULL;

} 

int main(void) 
{ 

	pthread_t threads[thread_count];
    // pthread_t thread_id; 
    // printf("Before Thread\n"); 
    for(int i = 0; i < thread_count; i++)
	{
	    int *arg = (int *) malloc(sizeof(*arg));
	    *arg = i;
	    int error = pthread_create(&(threads[i]), NULL, multiThread_Handler, arg);
	    
	    if (error != 0)
	                printf("\nCan't create thread :[%s]", strerror(error));
	}

	for(int i = 0; i< thread_count; i++) {
		pthread_join(threads[i], NULL);
	}

    // pthread_create(&thread_id, NULL, myThreadFun, NULL); 
    // pthread_join(thread_id, NULL); 
    // printf("After Thread\n"); 
    exit(0); 



	// int i = 0; 
	// int error; 

	// if (pthread_mutex_init(&lock, NULL) != 0) 
	// { 
	// 	printf("\n mutex init has failed\n"); 
	// 	return 1; 
	// } 

	// while(i < 2) 
	// { 
	// 	error = pthread_create(&(tid[i]), NULL, &trythis, NULL); 
	// 	if (error != 0) 
	// 		printf("\nThread can't be created :[%s]", strerror(error)); 
	// 	i++; 
	// } 

	// pthread_join(tid[0], NULL); 
	// pthread_join(tid[1], NULL); 
	// pthread_mutex_destroy(&lock); 

	// return 0; 
} 
