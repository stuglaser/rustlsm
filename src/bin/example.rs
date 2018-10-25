#![allow(dead_code, unused)]
extern crate rustlsm;

use std::env;
use std::path::Path;

extern crate rand;
use rand::{Rng, SeedableRng, IsaacRng};

use rustlsm::{LsmTree, Options};

// Borrowed from seed_from_u64, which will be in rand 0.5
fn make_seeded_rng<T: SeedableRng>(mut state: u64) -> T {
    use std::ptr::copy_nonoverlapping;
    // We use PCG32 to generate a u32 sequence, and copy to the seed
    const MUL: u64 = 6364136223846793005;
    const INC: u64 = 11634580027462260723;

    let mut seed = T::Seed::default();
    for chunk in seed.as_mut().chunks_mut(4) {
        // We advance the state first (to get away from the input value,
        // in case it has low Hamming Weight).
        state = state.wrapping_mul(MUL).wrapping_add(INC);

        // Use PCG output function with to_le to generate x:
        let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
        let rot = (state >> 59) as u32;
        let x = xorshifted.rotate_right(rot).to_le();

        unsafe {
            let p = &x as *const u32 as *const u8;
            copy_nonoverlapping(p, chunk.as_mut_ptr(), chunk.len());
        }
    }

    T::from_seed(seed)
}

fn make_random_keys<R: Rng>(rng: &mut R, n: usize) -> Vec<String> {
    let mut keys : Vec<String> = Vec::new();
    for _ in 0..n {
        let s : String = rng
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(12).collect();
        //println!("S = {}", s);
        keys.push(s);
    }
    keys
}

const BIGDATA : &str = "abcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJabcdefghijABCDEFGHIJ";

pub fn main() {
    let args : Vec<String> = env::args().collect();

    // Random data
    let mut rng = make_seeded_rng::<IsaacRng>(53335);
    let mut keys = make_random_keys(&mut rng, 1000000);

    let mut tree = LsmTree::new(&Path::new(&args[1]), Options::default()).unwrap();
    tree.delete_unused_slab_files().unwrap();

    for (i, k) in keys.iter().enumerate() {
        //tree.set(&k, &format!("x_{}__{}", k, BIGDATA)).unwrap();
        tree.set(&k, &format!("x_{}", k)).unwrap();
        //thread::sleep(Duration::from_millis(100));
        if i % 1000 == 0 {
            print!(".");
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!("ok");
    tree.flush().unwrap();
    tree.dump_metadata();
}
