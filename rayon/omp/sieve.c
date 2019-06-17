#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#define MARKED 1
#define UNMARKED 0

int primes_serial(int n);
int primes_parallel(int n);

int main(int argc, char** argv)
{
    for(int ii = 2; ii < 8; ii++)
    {
        int n = pow(10, ii);

        double t1 = omp_get_wtime();
        int primes_ser = primes_serial(n);
        double t2 = omp_get_wtime();
        int primes_par = primes_parallel(n);
        double t3 = omp_get_wtime();

        printf("%d %d %d\n", primes_ser, primes_par, n);
        printf("Sequential: %f, Parallel: %f\n", t2 - t1, t3 - t2);
    }
    return 0;
}

int primes_serial(int n)
{
    char * slots = (char *) malloc(n * sizeof(char));

    for(int i = 0; i < n; i++)
    {
        slots[i] = UNMARKED;
    }

    int end = sqrt(n);

    double start_time = omp_get_wtime();
    for(int i = 2; i <= end; i++)
    {
        if(slots[i-1] == MARKED)
        {
            continue;
        }
        else
        {
            #pragma omp parallel for schedule(runtime)
            for(int j = i*i; j <= n; j += i)
            {
                // printf("%d %d\n", j, i);
                slots[j-1] = MARKED;
            }
        }
    }
    
    double end_time = omp_get_wtime();

    int num_primes = 0;
    for(int i = 0; i < n; i++)
    {
        if(slots[i] == UNMARKED)
        {
            num_primes++;
            // printf("%d %s\n", i + 1, "is prime");
        }
    }
    free(slots);
    // printf("%d\n", num_primes);
    return num_primes;
}

int primes_parallel(int n)
{
    char * slots = (char *) malloc(n * sizeof(char));
    for(int i = 0; i < n; i++)
    {
        slots[i] = UNMARKED;
    }

    int end = sqrt(n);

    for(int i = 2; i <= end; i++)
    {
        if(slots[i-1] == MARKED)
        {
            continue;
        }
        else
        {

            #pragma omp parallel for schedule(runtime)
            for(int j = i*i; j <= n; j += i)
            {
                slots[j-1] = MARKED;
            }
        }
    }

    int num_primes = 0;
    for(int i = 0; i < n; i++)
    {
        if(slots[i] == UNMARKED)
            num_primes++;
    }
    free(slots);
    return num_primes;
}