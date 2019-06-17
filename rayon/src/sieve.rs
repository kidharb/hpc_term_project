use rayon::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn sieve()
{
    for ii in 2..8
    {
        let n = 10usize.pow(ii);

        let t1 = since_the_epoch();
        let primes_ser = primes_serial(n);
        let t2 = since_the_epoch();
        let primes_par = primes_parallel(n);
        let t3 = since_the_epoch();

        println!("{} {} {}", primes_ser, primes_par, n);
        println!("Sequential: {}, Parallel: {}", t2 - t1, t3 - t2);
    }
}

const unmarked: bool = false;
const marked: bool = true;

fn primes_serial(n: usize) -> usize
{
    let mut slots: Vec<_> = (0..n).map(|x| unmarked).collect();
    let end = (n as f64).sqrt() as usize;

    for i in 2..(end+1)
    {
        if slots[i-1] == marked
        {
            continue;
        }
        else
        {
            for j in ((i*i)..n).step_by(i)
            {
                slots[j-1] = marked;
            }
        }
    }

    let num_primes = slots.into_iter().filter(|x| *x == unmarked).count();
    num_primes
}

fn primes_parallel(n: usize) -> usize
{
    let mut slots: Vec<_> = (0..n).map(|x| unmarked).collect();
    let end = (n as f64).sqrt().ceil() as usize;

    for i in 2..(end+1)
    {
        if slots[i-1] == marked
        {
            continue;
        }
        else
        {
            // for j in ((i*i)..n).step_by(i)
            (((i*i)..n).step_by(i))
            .collect::<Vec<usize>>()
            .into_par_iter()
            .for_each(|j|
            {
                slots[j-1] = marked;
            });
        }
    }

    let num_primes = slots.into_iter().filter(|x| *x == unmarked).count();
    num_primes
}

fn since_the_epoch() -> u64 {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH).expect(
        "Time went backwards",
    );
    since_the_epoch.as_secs() * 1000 + since_the_epoch.subsec_nanos() as u64 / 1_000_000
}
