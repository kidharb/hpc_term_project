use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn mat()
{
    for ii in 1..9
    {
        let a_rows = 320 * ii;
        let a_cols = 240;
        let a_size = a_rows * a_cols;
        let b_rows = a_cols;
        let b_cols = a_rows;
        let b_size = b_rows * b_cols;

        let mat_a = Matrix {
            data: (1..(a_size+1)).map(|a| a as f32).collect(),//vec![1., 2., 3.,   4., 5., 6.],
            cols: a_cols,
            rows: a_rows
        };
        let mat_b = Matrix {
            data: ((a_size+1)..(a_size + b_size + 1)).map(|a| a as f32).collect(),//vec![1., 2., 3.,   4., 5., 6.],
            cols: b_cols,
            rows: b_rows
        };

        let t1 = since_the_epoch();
        let res_serial = mat_serial(&mat_a, &mat_b);
        let t2 = since_the_epoch();
        let res_parallel = mat_par(&mat_a, &mat_b);
        let t3 = since_the_epoch();

        println!("Sequential: {}, Parallel: {}", t2 - t1, t3 - t2);

        // print_mat(&mat_a);
        // print_mat(&mat_b);
        // print_mat(&res_serial);
        // print_mat(&res_parallel);
    }
}

pub fn since_the_epoch() -> u64 {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH).expect(
        "Time went backwards",
    );
    since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000
}

pub struct Matrix
{
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize
}

fn mat_serial(a: &Matrix, b: &Matrix) -> Matrix
{
    let mut the_return = Matrix {
        rows: a.rows,
        cols: b.cols,
        data: vec![0.; a.rows * b.cols]
    };
    for a_row in 0..a.rows
    {
        for b_col in 0..b.cols
        {
            let mut sum = 0.;

            for cell_index in 0..a.cols
            {
                let a_val = a.data[cell_index + a_row * a.cols];
                let b_val = b.data[b_col + cell_index * b.cols];

                sum += a_val * b_val;
            }

            the_return.data[b_col + a_row * b.cols] = sum;
        }
    };
    the_return
}

fn mat_par(a: &Matrix, b: &Matrix) -> Matrix
{
    let rows = a.rows;
    let cols = b.cols;
    let total_length = rows * cols;

    let data: Vec<_> = (0..total_length)
        .into_par_iter()
        .map(|i| {
            let a_row = i / cols;
            let b_col = i % cols;

            let mut sum = 0.;

            for cell_index in 0..a.cols
            {
                let a_val = a.data[cell_index + a_row * a.cols];
                let b_val = b.data[b_col + cell_index * b.cols];

                sum += a_val * b_val;
            }
            sum
        })
        .collect();

    Matrix {
        rows,
        cols,
        data
    }
}

fn print_mat(mat: &Matrix)
{
    for row in 0..mat.rows
    {
        for col in 0..mat.cols
        {
            print!("{:2} ", mat.data[row * mat.cols + col]);
        }
        println!("");
    }
    println!("");
}
