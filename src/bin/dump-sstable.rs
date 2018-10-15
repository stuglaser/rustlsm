extern crate rustlsm;

use std::env;
use std::path::Path;

use rustlsm::SSTable;

pub fn main() {
    let args : Vec<String> = env::args().collect();

    let sstable = SSTable::open(&Path::new(&args[1])).unwrap();
    for item in sstable.iter() {
        let (key, value) = item.unwrap();
        println!("{}|{}", key, value);
    }
}
