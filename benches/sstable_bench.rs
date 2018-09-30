#[macro_use]
extern crate criterion;
use criterion::Criterion;

extern crate rand;
use rand::{Rng, SeedableRng, IsaacRng};
extern crate tempfile;
use tempfile::NamedTempFile;

extern crate rustlsm;
use rustlsm::{SSTable, SSTableBuilder};

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

fn criterion_benchmark(c: &mut Criterion) {
    // Random data
    let mut rng = make_seeded_rng::<IsaacRng>(53335);
    let mut keys = make_random_keys(&mut rng, 1000);
    keys.sort();

    // Creates the SSTable
    let file = NamedTempFile::new().unwrap();

    let mut builder = SSTableBuilder::create(file.path()).unwrap();
    for k in &keys {
        builder.add(k, &format!("xx_{}", k)).unwrap();
    }
    builder.finish().unwrap();
    drop(builder);

    let mut sstable = SSTable::open(file.path());
    let n = (keys.len() as f32 * 0.9) as usize;
    c.bench_function("sstable.get", move |b| b.iter(|| sstable.get(&keys[n]).unwrap()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
