#include <stdio.h>
#include <omp.h>
#include <math.h>

double pi_parallel(double step_size, long);
double pi_serial(double step_size, long);

int main(int argc, char**argv)
{
    for(int ii = 2; ii < 10; ii++)
    {
        long num_steps = pow(10, ii);//1000000;
        double step_size = 1.0 / (double)num_steps;
        // double pi_par = pi_parallel(step_size);

        double t1 = omp_get_wtime();
        double pi_ser = pi_serial(step_size, num_steps);
        double t2 = omp_get_wtime();
        double pi_par = pi_parallel(step_size, num_steps);
        double t3 = omp_get_wtime();

        printf("%f %f %d\n", pi_ser, pi_par, num_steps);
        printf("Sequential: %f, Parallel: %f\n", t2 - t1, t3 - t2);
    }
}

double pi_parallel(double step_size, long num_steps)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for(int i = 1; i <= num_steps; i++)
    {
        double x = (i - 0.5) * step_size;
        sum = sum + 4.0/(1.0+x*x);
    }
    double pi = step_size * sum;
    return pi;
}

double pi_serial(double step_size, long num_steps)
{
    double sum = 0.0;
    for(int i = 1; i <= num_steps; i++)
    {
        double x = (i - 0.5) * step_size;
        sum = sum + 4.0/(1.0+x*x);
    }
    double pi = step_size * sum;
    return pi;
}
