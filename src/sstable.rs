extern crate byteorder;
extern crate rand;
extern crate tempfile;
extern crate uuid;
extern crate owning_ref;

use std::collections::HashMap;
use std::fs;
use std::io;
use std::io::{Read, Write, BufReader, BufWriter, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::str;

use byteorder::*;
use byteorder::LittleEndian as Endian;
use owning_ref::OwningHandle;


const SSTABLE_SIGNATURE : &'static str = "SSTB";
const SSTABLE_RW_BUFFER_SIZE : usize = 32*1024;

pub struct SSTable {
    path: PathBuf,
    size: u64,
    index: HashMap<String, u64>,
}

pub struct SSTableIter<'a> {
    file: BufReader<fs::File>,
    sstable: &'a SSTable,
    loc: u64,
}

// Reads a record at the current location
fn read_record<R: Read>(file: &mut R) -> Result<(String, String), io::Error> {
    let keylen = file.read_u32::<Endian>()? as usize;
    let valuelen = file.read_u32::<Endian>()? as usize;

    let mut keybuf = vec![0 as u8; keylen];
    file.read_exact(&mut keybuf)?;
    let key = String::from_utf8(keybuf).unwrap();

    let mut valuebuf = vec![0 as u8; valuelen];
    file.read_exact(&mut valuebuf)?;
    let value = String::from_utf8(valuebuf).unwrap();

    Ok((key, value))
}

fn write_record<W: Write>(file: &mut W, key: &str, value: &str)
                -> Result<usize, io::Error> {
    let keybytes = key.as_bytes();
    let valuebytes = value.as_bytes();
    file.write_u32::<Endian>(keybytes.len() as u32)?;
    file.write_u32::<Endian>(valuebytes.len() as u32)?;
    file.write_all(keybytes)?;
    file.write_all(valuebytes)?;
    Ok(4 + 4 + keybytes.len() + valuebytes.len())
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
        Some(read_record(&mut self.file))
    }
}

impl<'a> ExactSizeIterator for SSTableIter<'a> {}

impl SSTable {
    pub fn open(path: &Path) -> Result<SSTable, io::Error> {
        let mut sstable = SSTable{ path: path.to_path_buf(),
                                   size: 0,
                                   index: HashMap::new() };
        sstable.preload()?;
        Ok(sstable)
    }

    fn preload(&mut self) -> Result<(), io::Error> {
        let mut file = BufReader::with_capacity(SSTABLE_RW_BUFFER_SIZE,
                                                fs::File::open(&self.path)?);

        // Reads the footer
        file.seek(SeekFrom::End(-20))?;

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

    pub fn iter<'a>(&'a self) -> SSTableIter {
        let file = BufReader::with_capacity(
            SSTABLE_RW_BUFFER_SIZE,
            fs::File::open(&self.path).unwrap());
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
        file.seek(SeekFrom::Start(loc))?;
        let (_key, value) = read_record(&mut file)?;
        Ok(Some(value))
    }

    pub fn len(&mut self) -> usize {
        self.size as usize
    }
}

pub struct SSTableBuilder {
    file: Box<BufWriter<fs::File>>,  // TODO: unnecessarily specific types
    index: Vec<(String, u64)>,
    bytes_written: usize,
    estimated_total_bytes: usize,
    finished: bool,
}

impl SSTableBuilder {
    pub fn create(path: &Path) -> Result<SSTableBuilder, io::Error> {
        let writer = Box::new(
            BufWriter::with_capacity(SSTABLE_RW_BUFFER_SIZE,
                                     fs::File::create(path)?));
        Ok(SSTableBuilder{
            file: writer,
            index: Vec::new(),
            bytes_written: 0,
            estimated_total_bytes: 0,
            finished: false})
    }

    pub fn add(&mut self, key: &str, value: &str) -> Result<(), io::Error> {
        assert!(!self.finished);
        let loc = self.bytes_written as u64;

        self.index.push((key.to_string(), loc));
        let bytes = write_record(&mut self.file, key, value)?;
        self.bytes_written += bytes;
        self.estimated_total_bytes += bytes + 4 + key.len() + 8;
        Ok(())
    }

    pub fn estimated_total_size(&self) -> usize {
        self.estimated_total_bytes
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

        self.file.seek(SeekFrom::Current(0))?;

        // Writes the footer
        self.file.write_all(SSTABLE_SIGNATURE.as_bytes())?;
        self.file.write_u64::<Endian>(self.index.len() as u64)?;
        self.file.write_u64::<Endian>(index_loc as u64)?;
        self.file.flush()?;

        self.finished = true;
        Ok(())
    }

    pub fn key_bounds(&self) -> (&str, &str) {
        (&self.index[0].0, &self.index.last().unwrap().0)
    }
}

impl Drop for SSTableBuilder {
    fn drop(&mut self) {
        if !self.finished {
            self.finish().unwrap();
        }
    }
}

/// Holds an SSTable and an iter to it together
type SSTableIterHandle = OwningHandle<Box<SSTable>, Box<SSTableIter<'static>>>;

/// For iterating through a list of SSTable's
pub struct SSTableChainer {
    sources: Vec<PathBuf>,
    idx: usize,
    current_iter: Option<SSTableIterHandle>,
}

impl SSTableChainer {
    pub fn new(sources: Vec<PathBuf>) -> Result<Self, io::Error> {
        let oh = if sources.len() == 0 {
            None
        }
        else {
            Some(SSTableChainer::iter_on_file(&sources[0])?)
        };
        Ok(SSTableChainer{
            sources: sources,
            idx: 0,
            current_iter: oh,
        })
    }

    fn iter_on_file(path: &PathBuf) ->
        Result<SSTableIterHandle, io::Error>
    {
        let sstable = Box::new(SSTable::open(path)?);
        Ok(OwningHandle::new_with_fn(
            sstable,
            |s| unsafe { Box::new((*s).iter()) } ))
    }
}

impl Iterator for SSTableChainer {
    type Item = Result<(String, String), io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let item = match self.current_iter {
                None => return None,
                Some(ref mut iter) => iter.next(),
            };

            if item.is_some() {
                // TODO: probably want to force ending this iteration on error
                return item;
            }

            // Advance
            self.idx += 1;
            self.current_iter = {
                if self.idx == self.sources.len() {
                    None
                }
                else {
                    match Self::iter_on_file(&self.sources[self.idx]) {
                        Ok(handle) => Some(handle),
                        Err(err) => return Some(Err(err)),
                    }
                }
            };
        }
    }
}

#[cfg(test)]
mod test {
    use {SSTable, SSTableBuilder};
    use tempfile::NamedTempFile;
    use rand::{Rng, SeedableRng, IsaacRng, self};

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
        builder.finish().unwrap();

        let mut sstable = SSTable::open(file.path()).unwrap();
        for k in &keys {
            assert_eq!(sstable.get(k).unwrap(), Some(format!("xx_{}", k)));
        }
    }

    #[test]
    fn sstable_key_not_present() {
        // Creates the SSTable
        let file = NamedTempFile::new().unwrap();
        let mut builder = SSTableBuilder::create(file.path()).unwrap();
        builder.add("bar", "xbar").unwrap();
        builder.add("foo", "xfoo").unwrap();
        builder.finish().unwrap();

        let mut sstable = SSTable::open(file.path()).unwrap();
        assert_eq!(sstable.get("abracadabra").unwrap(), None);
    }
}
