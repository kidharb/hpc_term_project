#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void mat_serial(
    float * a_data, int a_rows, int a_cols,
    float * b_data, int b_rows, int b_cols,
    float * res_data, int* res_rows, int* res_cols);
void mat_parallel(
    float * a_data, int a_rows, int a_cols,
    float * b_data, int b_rows, int b_cols,
    float * res_data, int* res_rows, int* res_cols);
void print_mat(float * data, int rows, int cols);

int main(int argc, char** argv)
{
    for(int ii = 1; ii < 9; ii++)
    {
        int a_cols = 320 * ii;
        int a_rows = 240;
        float* a_data = (float*)malloc(a_cols * a_rows * sizeof(float));
        int a_size = a_cols * a_rows;
        for(int i = 0; i < a_size; i++)
        {
            a_data[i] = i + 1;
        }
        
        int b_cols = a_rows;
        int b_rows = a_cols;
        float* b_data = (float*)malloc(b_cols * b_rows * sizeof(float));
        int b_size = b_cols * b_rows;
        for(int i = 0; i < b_size; i++)
        {
            b_data[i] = i + a_size + 1;
        }
        
        int res_cols = 0;
        int res_rows = 0;
        float* res_data = (float*)malloc(a_rows * b_cols * sizeof(float));
        
        int res_cols_2 = 0;
        int res_rows_2 = 0;
        float* res_data_2 = (float*)malloc(a_rows * b_cols * sizeof(float));


        double t1 = omp_get_wtime();
        mat_serial(a_data, a_rows, a_cols, b_data, b_rows, b_cols, res_data, &res_rows, &res_cols);
        double t2 = omp_get_wtime();
        mat_parallel(a_data, a_rows, a_cols, b_data, b_rows, b_cols, res_data_2, &res_rows_2, &res_cols_2);
        double t3 = omp_get_wtime();

        printf("Sequential: %f, Parallel: %f\n", t2 - t1, t3 - t2);

        // print_mat(a_data, a_rows, a_cols);
        // print_mat(b_data, b_rows, b_cols);
        // print_mat(res_data, res_rows, res_cols);
        // print_mat(res_data_2, res_rows_2, res_cols_2);

        free(a_data);
        free(b_data);
        free(res_data);
        free(res_data_2);
    }

    return 0;
}

void mat_serial(
    float * a_data, int a_rows, int a_cols,
    float * b_data, int b_rows, int b_cols,
    float * res_data, int* res_rows, int* res_cols)
{
    *res_rows = a_rows;
    *res_cols = b_cols;
    
    for(int a_row = 0; a_row < a_rows; a_row++)
    {
        for(int b_col = 0; b_col < b_cols; b_col++)
        {
            float sum = 0.;

            for (int cell_index = 0; cell_index < a_cols; cell_index++)
            {
                float a_val = a_data[cell_index + a_row * a_cols];
                float b_val = b_data[b_col + cell_index * b_cols];

                sum += a_val * b_val;
            }

            res_data[b_col + a_row * b_cols] = sum;
        }
    }
}

void mat_parallel(
    float * a_data, int a_rows, int a_cols,
    float * b_data, int b_rows, int b_cols,
    float * res_data, int* res_rows, int* res_cols)
{
    *res_rows = a_rows;
    *res_cols = b_cols;
    
    #pragma omp parallel for
    for(int a_row = 0; a_row < a_rows; a_row++)
    {
        for(int b_col = 0; b_col < b_cols; b_col++)
        {
            float sum = 0.;

            for (int cell_index = 0; cell_index < a_cols; cell_index++)
            {
                float a_val = a_data[cell_index + a_row * a_cols];
                float b_val = b_data[b_col + cell_index * b_cols];

                sum += a_val * b_val;
            }

            res_data[b_col + a_row * b_cols] = sum;
        }
    }
}

void print_mat(float * data, int rows, int cols)
{
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            printf("%.0f ", data[row * cols + col]);
        }
        printf("\n");
    }
    printf("\n");
}
