// EVERYTHING IS COLLUMN MAJOR

extern crate itertools;
extern crate itertools_num;
extern crate blas;
extern crate openblas_src;

use itertools_num::linspace;
use std::convert::TryInto;
use rand::seq::SliceRandom;
use rand::Rng;
use blas::*;


struct Mesh {
    n: usize,         // Dimentionality
    df: f64,     // Frame size parameter (0, inf)
    dm: f64,        // Mesh size parameter (0, inf)
    tau: f64,       // Mesh size adjustment parameter (0,1: rational)
    l_c: usize,
}
impl Mesh {
    fn create() -> Mesh {
        Mesh {
            n: 3,
            df: 1.0/16.0,
            dm: 1.0/16.0,
            tau: 4.,
            l_c: 0,
        }
    }
}

struct MeshPoint {
    x: Vec<f64>,    // Location of point (size = [n])
    y: f64,         // Function evaluation
}

fn main() {
    println!("Hello, beginning SRuMADS!");

    println!("Initialising optimiser framework...");

    let mesh = Mesh::create();

    poll_points(&mesh);





    

}

fn black_box(x: &Vec<f64>) -> f64 {
    x[0] + x[1]
}

fn rand_vector(mesh: &Mesh) -> (Vec<f64>, usize) {

    let l = -((mesh.dm as f64).log(mesh.tau)) as usize;
    let lim = (2.0 as f64).powf(l as f64);
    let i_hat = rand::thread_rng().gen_range(0, mesh.n);
    //let f = (mesh.dm / mesh.ds) as f64;      // Set upper and lower limits
    let len = (1 + (lim as i32 -1) - (-lim as i32 +1)) as usize;             // Find length of possible vector space
    
    let axis: Vec<_> = linspace::<f64>(-lim +1., lim -1., len)
                        .collect();
    println!("{:?}", axis);
    let mut b = vec![(0.0 as f64).powf(l as f64); mesh.n];
    for i in 0..mesh.n {
        b[i] = *axis
                .choose(&mut rand::thread_rng())
                .expect("axis.choose didnt work");
    }
    b[i_hat] = *[lim, -lim]
                .choose(&mut rand::thread_rng())
                .expect("lim choose didnt work");
    println!("print b{:?}", b);
    (b, i_hat)
}

fn poll_directions(mesh: &Mesh) -> Vec<f64> {
    let l = -((mesh.dm as f64).log(mesh.tau)) as usize;
    let lim = (2.0 as f64).powf(l as f64);
    let len = (1 + (lim as i32 -1) - (-lim as i32 +1)) as usize;             // Find length of possible vector space
    
    let axis: Vec<_> = linspace::<f64>(-lim +1., lim -1., len)
                        .collect();


    let res = rand_vector(&mesh);
    let mut b: Vec<f64> = res.0;
    let i_hat: usize = res.1;
    println!("{}", &i_hat);
    // Make L
    let mut L = vec![0.0; (mesh.n-1).pow(2)];
    for i in 0..mesh.n-1 {
        L[i * (mesh.n-1) + i] = *[lim, -lim]
                                .choose(&mut rand::thread_rng())
                                .expect("lim choose didnt work");
        for j in i+1..mesh.n-1 {
            L[i * (mesh.n-1) + j] = *axis
                                    .choose(&mut rand::thread_rng())
                                    .expect("axis.choose didnt work");
        }
    }
    println!("L{:?}", L);

    // Make B
    let mut p: Vec<usize> = (0..mesh.n).collect();
    p.remove(i_hat);
    p.shuffle(&mut rand::thread_rng());
    let mut B = vec![0.0; mesh.n*(mesh.n-1)];
    let mut r = 0;
    
    for c in 0..mesh.n-1 {
        r = 0;
        for p_r in &p {
            //println!("{}, {}", p_r, r);
            B[p_r + (mesh.n) * c] = L[r + (mesh.n-1) * c]; 
            r = r + 1;
        }
       
    }
    B.append(&mut b);
    println!("B{:?}", &B);

    //Make B_prime
    let mut q: Vec<usize> = (0..mesh.n).collect();
    let mut B_prime = vec![0.0; mesh.n.pow(2)];
    q.shuffle(&mut rand::thread_rng());
    let mut c = 0;
    for q_c in q {
        //println!("{}, {}", q_i, i);
        for r in 0..mesh.n {
            //println!("{}, {}", q_i * mesh.n + j, i * mesh.n + j);
            B_prime[q_c * mesh.n + r] = B[c * mesh.n + r];
        }
        c = c + 1;
    }
    println!("B_p{:?}", B_prime);

    let min = true;
    let mut D: Vec<f64>;
    if min == true {
        let mut d = vec![0.0; mesh.n];
        for r in 0..mesh.n {
            for c in 0..mesh.n {
                d[r] = d[r] - B_prime[c * mesh.n + r]
            }
        }
        println!("d{:?}", d);
        B_prime.append(&mut d);
        D = B_prime.to_vec();
        println!("D{:?}", &D);
    }
    else {
        let mut B_prime_neg = B_prime.iter()
                                     .map(|x| x * -1.0)
                                     .collect();
        B_prime.append(&mut B_prime_neg);
        D = B_prime.to_vec();
        println!("D{:?}", &D);
    }
    D
}
    
fn poll_points(mesh: &Mesh) {
    let x = vec![0.0; mesh.n];
    let P: Vec<f64> = poll_directions(&mesh).chunks(mesh.n)
                                  .map(|i| i[0] * mesh.dm)
                                  .collect();
    println!("newD{:?}", &P);                          


}
    
    



