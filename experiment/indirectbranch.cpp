#include<stdio.h> 
#include<string.h> 
#include<pthread.h> 
#include<stdlib.h> 
#include<unistd.h> 

pthread_mutex_t lock; 
int list_size = 100000000;
int thread_count;
int array[list_size];

// Recall that an indirect branch is one for which the target must be determined 
// at run-time; common examples being polymorphic code or a jump-table.  
// In the example above, when the virtual method Foo() is invoked, dynamic lookup 
// must be performed to determine the location of the matching implementation.  
// In this case, rather than stalling speculative execution until this target 
// can be determined, the hardware may try to guess.

class Base {
	public:
  	virtual int Foo (int a) {
  		return a + 1;
  	};
};

class Derived : public Base {
	public:
  	int Foo (int a){
  		return a + 2;
  	};
};

void* multiThread_Handler (void *arg) 
{ 
	  int thread_index = *((int *)arg);

   	pthread_mutex_lock(&lock); 
   	unsigned int start_index = (thread_index*(list_size/thread_count));
   	unsigned int end_index = ((thread_index+1) * (list_size/thread_count));
   	pthread_mutex_unlock(&lock); 

   	for(int i =  start_index; i < end_index;  i++)
   	{
   		array[i] = array[i] + 1;
   		Base* obj = new Derived;
		array[i] = obj->Foo(array[i]);
   	}
   	return NULL;
} 

int main(int argc, char *argv[]) 
{
  thread_count = atoi(argv[1]);
	pthread_t threads[thread_count];
    	for (int i = 0; i < thread_count; i++)
	{
		int *arg = (int *) malloc(sizeof(*arg));
	    	*arg = i;
	    	int error = pthread_create(&(threads[i]), NULL,
	    	    multiThread_Handler, arg);
		if (error != 0)
		{
	                printf("\nCan't create thread :[%s]", strerror(error));
		}
	}

	for (int i = 0; i< thread_count; i++) {
		pthread_join(threads[i], NULL);
	}

    exit(0); 
} 
