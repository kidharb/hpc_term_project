use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn pi()
{
    for ii in 2..10
    {
        let num_steps: usize = 10_usize.pow(ii);//1000000;
        let step_size = 1.0 / num_steps as f64;

        // let pi_ser = pi_serial(step_size, num_steps);
        let t1 = since_the_epoch();
        let pi_ser = pi_serial(step_size, num_steps);
        let t2 = since_the_epoch();
        let pi_par = pi_parallel(step_size, num_steps);
        let t3 = since_the_epoch();

        println!("{} {} {}", pi_ser, pi_par, num_steps);

        println!("Sequential: {}, Parallel: {}", t2 - t1, t3 - t2);
    }
}

fn pi_parallel(step_size: f64, num_steps: usize) -> f64
{
    // let mut sum = 0.0;
    
    let sum: f64 = (1..(num_steps+1))
        .into_par_iter()
        // .map(|x| x as f64)
        .into_par_iter()
        .fold_with(0., |sum: f64, i: usize|
        {
            let x = (i as f64 - 0.5) * step_size;
            sum + 4.0/(1.0+x*x)
        })
        .sum();
    let pi = step_size * sum;
    // print!("{:?}", sum);
    pi
}

fn pi_serial(step_size: f64, num_steps: usize) -> f64
{
    let mut sum = 0.0;
    //for(int i = 1; i <= num_steps; i++)
    for i in 1..(num_steps+1)
    {
        let x = (i as f64 - 0.5) * step_size;
        sum = sum + 4.0/(1.0+x*x);
    }
    let pi = step_size * sum;
    pi
}

fn since_the_epoch() -> u64 {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH).expect(
        "Time went backwards",
    );
    since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000
}
