#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

float* sort_serial(float* rows, int count);
float* sort_parallel(float* rows, int count);
void swap(float* rows, int a, int b);
void q_quick_sort(float* arr, int low, int high);

int main(int argc, char** argv)
{

    for(int ii = 1; ii < 9; ii++)
    {
        int size = 100000 * ii;
        float* data = (float*)malloc(size * sizeof(float));
        float* data2 = (float*)malloc(size * sizeof(float));
        srand(time(NULL));
        for(int i = 0; i < size; i++)
        {
            data2[i] = data[i] = rand();
        }


        double t1 = omp_get_wtime();
        sort_serial(data, size);
        double t2 = omp_get_wtime();
        sort_parallel(data2, size);
        double t3 = omp_get_wtime();

        printf("Sequential: %f, Parallel: %f\n", t2 - t1, t3 - t2);

        free(data);
        free(data2);
    }

    return 0;
}

void q_qsort_p(float* arr, int low, int high)
{
    if(low < high)
    {
        int pi = q_partition(arr, low, high);
  
        q_quick_sort(arr, low, pi - 1);
        q_quick_sort(arr, pi + 1, high);
    } 
}

float* sort_serial(float* rows, int count)
{
    q_qsort_p(rows, 0, count-1);
    return rows;
}

float* sort_parallel(float* rows, int count)
{
    q_quick_sort(rows, 0, count-1);
    return rows;
}

int q_partition(float* arr, int low, int high) 
{ 
    float pivot = arr[high];
    int i = low - 1;
  
    for (int j = low; j <= high- 1; j++) 
    { 
        if (arr[j] <= pivot) 
        { 
            i++;
            swap(arr, i, j); 
        } 
    } 
    swap(arr, i + 1, high); 
    return (i + 1);
} 

void q_quick_sort(float* arr, int low, int high)
{
    if(low < high)
    {
        int pi = q_partition(arr, low, high); 
  
        #pragma omp parallel
        #pragma omp sections
        {
            #pragma omp section
            {
                q_quick_sort(arr, low, pi - 1); 
            }
            #pragma omp section
            {
                q_quick_sort(arr, pi + 1, high);
            }
        }
    } 
}

void swap(float* rows, int a, int b)
{
    float temp = rows[a];
    rows[a] = rows[b];
    rows[b] = temp;
}
