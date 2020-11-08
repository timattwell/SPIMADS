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
    delta: f64,     // Frame size parameter (0, inf)
    ds: usize,        // Mesh size parameter (0, inf)
    tau: f64,       // Mesh size adjustment parameter (0,1: rational)
    l_c: usize,
}
impl Mesh {
    fn create() -> Mesh {
        Mesh {
            n: 5,
            delta: 1.0/16.0,
            ds: 1,
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

    poll_directions(&mesh);





    

}

fn black_box(x: &Vec<f64>) -> f64 {
    x[0] + x[1]
}

fn rand_vector(mesh: &Mesh) -> (Vec<f64>, usize) {

    let l = -((mesh.delta as f64).log(mesh.tau)) as usize;
    let lim = (2.0 as f64).powf(l as f64);
    let i_hat = rand::thread_rng().gen_range(0, mesh.n);
    //let f = (mesh.delta / mesh.ds) as f64;      // Set upper and lower limits
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

fn poll_directions(mesh: &Mesh) {
    let l = -((mesh.delta as f64).log(mesh.tau)) as usize;
    let lim = (2.0 as f64).powf(l as f64);
    let len = (1 + (lim as i32 -1) - (-lim as i32 +1)) as usize;             // Find length of possible vector space
    
    let axis: Vec<_> = linspace::<f64>(-lim +1., lim -1., len)
                        .collect();


    let res = rand_vector(&mesh);
    let b: Vec<f64> = res.0;
    let i_hat: usize = res.1;

    let mut L = vec![0.0; (mesh.n-1).pow(2)];
    for i in 0..mesh.n-1 {
        L[i * (mesh.n-1) + i] = *[lim, -lim]
                                .choose(&mut rand::thread_rng())
                                .expect("lim choose didnt work");
        for j in i+1..mesh.n-1 {
            L[i * (mesh.n-1) + j] = *N
                                    .choose(&mut rand::thread_rng())
                                    .expect("axis.choose didnt work");
        }
    }
    let mut p: Vec<u32> = (0..mesh.n)
                            .collect()
                            .shuffle(&mut thread_rng());
    let mut B = vec![0.0; (mesh.n-1).pow(2)];
    let i = 0;
    for p_i in p {
        for j in 0..mesh.n {
            B[p_i * mesh.n + j] = L[i * (mesh.n-1) +j]
        }

    }
    println!("{:?}", L);
    let mut h2: Vec<f64> = Vec::new();
    unsafe {
        h2.push(ddot(mesh.n.try_into().unwrap(), &v, 1, &v, 1));
        daxpy(mesh.n.pow(2).try_into().unwrap(), -2.0, &h2, 0, &mut L, 1);
    }
    println!("{:?}", L);
    // Now have the householder matrix
    


}
