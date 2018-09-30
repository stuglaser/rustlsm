#![allow(dead_code, unused)]

extern crate tempfile;
extern crate byteorder;
extern crate rand;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::io;
use std::fs;
use std::io::{Read, Write, Seek, SeekFrom};
use std::io::ErrorKind::UnexpectedEof;
use std::cmp::max;
use std::str;
use std::marker::PhantomData;

use byteorder::*;

use byteorder::LittleEndian as Endian;


const SSTABLE_SIGNATURE : &'static str = "SSTB";

pub struct SSTable {
    path: PathBuf,
    size: u64,
    index: HashMap<String, u64>,
}

pub struct SSTableIter<'a> {
    file: fs::File,
    sstable: &'a SSTable,
    loc: u64,
}

// Reads everything but keylen from the record. This is kind of silly, it just
// exists to help the error handling, so UnexpectedEof can be caught for just
// the first thing we read from the record.
fn read_rest_of_record(file: &mut fs::File, keylen: usize)
                       -> Result<(String, String), io::Error> {
    let valuelen = file.read_u32::<Endian>()? as usize;

    let mut keybuf = vec![0 as u8; keylen];
    file.read_exact(&mut keybuf)?;
    let key = String::from_utf8(keybuf).unwrap();

    let mut valuebuf = vec![0 as u8; valuelen];
    file.read_exact(&mut valuebuf)?;
    let value = String::from_utf8(valuebuf).unwrap();

    Ok((key, value))
}

// Reads a record.  Returns None on EOF
fn read_record(file: &mut fs::File) -> Option<Result<(String, String), io::Error>> {
    let keylen = match file.read_u32::<Endian>() {
        Ok(len) => len as usize,
        Err(err) => {
            if err.kind() == UnexpectedEof {
                return None
            }
            return Some(Err(err));
        }
    };

    Some(read_rest_of_record(file, keylen))
}

fn write_record(file: &mut fs::File, key: &str, value: &str)
                -> Result<(), io::Error> {
    let keybytes = key.as_bytes();
    let valuebytes = value.as_bytes();
    file.write_u32::<Endian>(keybytes.len() as u32)?;
    file.write_u32::<Endian>(valuebytes.len() as u32)?;
    file.write_all(keybytes)?;
    file.write_all(valuebytes)?;
    Ok(())
}

impl<'a> SSTableIter<'a> {
    pub fn done(&self) -> bool {
        self.loc >= self.sstable.size
    }
}

impl<'a> Iterator for SSTableIter<'a> {
    type Item = Result<(String, String), io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done() {
            return None
        }

        self.loc += 1;
        match read_record(&mut self.file) {
            None => {
                //self.done = true;
                //None
                panic!("I don't think this should happen anymore");
            },
            Some(Ok(x)) => Some(Ok(x)),
            Some(Err(err)) => {
                //self.done = true;
                self.loc = self.sstable.size;
                Some(Err(err))
            }
        }
    }
}

impl<'a> ExactSizeIterator for SSTableIter<'a> {}

impl SSTable {
    pub fn open(path: &Path) -> SSTable {
        let mut sstable = SSTable{ path: path.to_path_buf(),
                                   size: 0,
                                   index: HashMap::new() };
        sstable.preload().unwrap();
        sstable
    }

    fn preload(&mut self) -> Result<(), io::Error> {
        let mut file = fs::File::open(&self.path)?;

        // Reads the footer
        file.seek(SeekFrom::End(-20))?;
        let footer_loc = file.seek(SeekFrom::Current(0))?;
        println!("Reading footer from {}", footer_loc);

        // Reads the signature
        let mut signature = [0; 4];
        file.read_exact(&mut signature)?;
        if signature != SSTABLE_SIGNATURE.as_bytes() {
            panic!("Invalid sstable signature: {:?}", signature);
        }

        self.size = file.read_u64::<Endian>()?;
        let index_offset = file.read_u64::<Endian>()?;

        // Reads the index
        file.seek(SeekFrom::Start(index_offset))?;
        for _ in 0..self.size {
            let keylen = file.read_u32::<Endian>()? as usize;
            let mut keybuf = vec![0 as u8; keylen];
            file.read_exact(&mut keybuf)?;
            let key = String::from_utf8(keybuf).unwrap();

            let offset = file.read_u64::<Endian>()?;
            self.index.insert(key, offset);
        }

        Ok(())
    }

    pub fn iter<'a>(&'a mut self) -> SSTableIter {
        let mut file = fs::File::open(&self.path).unwrap();
        SSTableIter::<'a>{file: file,
                          sstable: self,
                          loc: 0}
    }

    pub fn get(&mut self, key: &str) -> Result<Option<String>, io::Error> {
        let loc = match self.index.get(key) {
            Some(loc) => *loc,
            None => return Ok(None),
        };

        let mut file = fs::File::open(&self.path)?;
        file.seek(SeekFrom::Start(loc));
        let (key, value) = read_record(&mut file).unwrap()?;
        Ok(Some(value))
    }

    pub fn len(&mut self) -> usize {
        self.size as usize
    }
}

pub struct SSTableBuilder {
    file: fs::File,
    index: Vec<(String, u64)>,
    finished: bool,
}

impl SSTableBuilder {
    pub fn create(path: &Path) -> Result<SSTableBuilder, io::Error> {
        Ok(SSTableBuilder{
            file: fs::File::create(path)?,
            index: Vec::new(),
            finished: false})
    }

    pub fn add(&mut self, key: &str, value: &str) -> Result<(), io::Error> {
        assert!(!self.finished);
        let loc = self.file.seek(SeekFrom::Current(0))?;
        self.index.push((key.to_string(), loc));
        write_record(&mut self.file, key, value)
    }

    pub fn finish(&mut self) -> Result<(), io::Error> {
        // Writes the index
        let index_loc = self.file.seek(SeekFrom::Current(0))?;
        for (key, loc) in &self.index {
            let keybytes = key.as_bytes();
            self.file.write_u32::<Endian>(keybytes.len() as u32)?;
            self.file.write_all(keybytes)?;
            self.file.write_u64::<Endian>(*loc)?;
        }

        let footer_loc = self.file.seek(SeekFrom::Current(0))?;
        println!("Writing footer to {}", footer_loc);

        // Writes the footer
        self.file.write_all(SSTABLE_SIGNATURE.as_bytes())?;
        self.file.write_u64::<Endian>(self.index.len() as u64)?;
        self.file.write_u64::<Endian>(index_loc as u64)?;

        self.finished = true;
        Ok(())
    }
}

impl Drop for SSTableBuilder {
    fn drop(&mut self) {
        if !self.finished {
            self.finish();
        }
    }
}

pub struct LsmTree {
    path: PathBuf,
    map: BTreeMap<String, String>
}

enum Which {
    MemNext,
    DiskNext,
    BothNext,
}

impl LsmTree {
    pub fn new(path: &Path) -> LsmTree {
        return LsmTree{ path: path.to_path_buf(), map: BTreeMap::new() };
    }
    fn set(&mut self, key: &str, value: &str) {
        self.map.insert(key.to_string(), value.to_string());
    }

    fn get(&self, key: &str) -> Option<String> {
        if let Some(s) = self.map.get(key) {
            return Some(s.to_string());
        }

        // Searches through the DB file
        let dbpath = self.path.join("records.db");
        if dbpath.exists() {
            let mut sstable = SSTable::open(&dbpath);
            return sstable.get(key).unwrap();  // TODO: unwrap
        }

        // key wasn't found anywhere
        None
    }

    pub fn compact(&mut self) -> Result<(), io::Error> {
        let dbpath = self.path.join("records.db");
        if dbpath.exists() {
            let mut sstable = SSTable::open(&dbpath);

            // Merges the mem table with the disk table
            let temppath = self.path.join("temp_records.db");
            //let mut file = fs::File::create(&temppath)?;
            let mut builder = SSTableBuilder::create(&temppath)?;

            let mut iter_mem = self.map.iter().peekable();
            let mut iter_disk = sstable.iter().peekable();
            loop {
                use self::Which::*;
                let which = match (iter_mem.peek(), iter_disk.peek()) {
                    (None, None) => break,
                    (Some(_), None) => MemNext,
                    (None, Some(_)) => DiskNext,
                    (_, Some(Err(_))) => DiskNext, // Force disk error below
                    (Some(mem), Some(Ok(disk))) => {
                        if mem.0 == &disk.0 { BothNext }
                        else if mem.0 < &disk.0 { MemNext }
                        else { DiskNext }
                    },
                };

                match which {
                    MemNext => {
                        let record = iter_mem.next().unwrap();
                        builder.add(&record.0, &record.1)?;
                    },
                    DiskNext => {
                        let record = iter_disk.next().unwrap()?;
                        builder.add(&record.0, &record.1)?;
                    },
                    BothNext => {
                        let record = iter_mem.next().unwrap();
                        builder.add(&record.0, &record.1)?;
                        iter_disk.next().unwrap();  // Drop
                    },
                }
            }
            builder.finish()?;
            drop(builder);
            fs::rename(temppath, dbpath)?;
        }
        else {
            let mut builder = SSTableBuilder::create(&dbpath)?;
            for (key, value) in self.map.iter() {
                builder.add(key, value)?;
            }
            builder.finish();
        }

        self.map.clear();
        Ok(())
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use {LsmTree, SSTable, SSTableBuilder};
    use tempfile::{NamedTempFile, Builder as TempDirBuilder, TempDir};
    use std::io;
    use rand::{Rng, SeedableRng, IsaacRng, self};

    fn create_temp_db() -> Result<(TempDir, LsmTree), io::Error> {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir()?;
        let tree = LsmTree::new(tmp_dir.path());
        println!("Created temp dir {:?}", tmp_dir);
        return Result::Ok((tmp_dir, tree));
    }

    #[test]
    fn sanity_single_put_get() {
        let (dir, mut tree) = create_temp_db().unwrap();
        tree.set("foo", "bar");
        assert_eq!(tree.get("foo"), Some("bar".to_string()));
    }

    #[test]
    fn single_value_compact() -> Result<(), io::Error> {
        let (dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "bar");
        tree.compact().unwrap();
        assert_eq!(tree.get("foo"), Some("bar".to_string()));
        Ok(())
    }

    #[test]
    fn double_compact() {
        let (dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "valfoo");
        tree.compact().unwrap();
        tree.set("bar", "valbar");

        assert_eq!(tree.get("foo"), Some("valfoo".to_string()), "after 1 compact");
        assert_eq!(tree.get("bar"), Some("valbar".to_string()), "after 1 compact");

        tree.compact().unwrap();
        assert_eq!(tree.get("foo"), Some("valfoo".to_string()), "after 2 compacts");
        assert_eq!(tree.get("bar"), Some("valbar".to_string()), "after 2 compacts");

        dir.close().unwrap();
    }

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

    #[test]
    fn sstable_multi() {
        // Random data
        let mut rng = make_seeded_rng::<IsaacRng>(53335);
        let mut keys = make_random_keys(&mut rng, 10);
        keys.sort();

        // Creates the SSTable
        let file = NamedTempFile::new().unwrap();

        let mut builder = SSTableBuilder::create(file.path()).unwrap();
        for k in &keys {
            builder.add(k, &format!("xx_{}", k)).unwrap();
        }
        builder.finish();

        let mut sstable = SSTable::open(file.path());
        for k in &keys {
            assert_eq!(sstable.get(k).unwrap(), Some(format!("xx_{}", k)));
        }
    }

    #[test]
    fn sstable_key_not_present() {
        // Creates the SSTable
        let file = NamedTempFile::new().unwrap();
        let mut builder = SSTableBuilder::create(file.path()).unwrap();
        builder.add("bar", "xbar");
        builder.add("foo", "xfoo");
        builder.finish();

        let mut sstable = SSTable::open(file.path());
        assert_eq!(sstable.get("abracadabra").unwrap(), None);
    }

    #[test]
    fn check_size() {
        let letters = "abcdefg";

        // Creates the SSTable
        let file = NamedTempFile::new().unwrap();

        let mut builder = SSTableBuilder::create(file.path()).unwrap();
        for ch in letters.chars() {
            let kv = format!("{}", ch);
            builder.add(&kv, &kv).unwrap();
        }
        builder.finish();

        let mut sstable = SSTable::open(file.path());
        assert_eq!(sstable.len(), letters.len());
    }
}
