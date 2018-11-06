#![allow(dead_code, unused)]

use std::fs;
use std::io;
use std::io::{Read, Write, BufReader, BufWriter, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use byteorder::*;
use byteorder::LittleEndian as Endian;

use super::varstring::{VarStringIORead, VarStringIOWrite};

type Result<T> = io::Result<T>;

const WAL_FILENAME : &str = "saved.wal";
const WAL_SIGNATURE : &str = "IWAL";

type WalRecord = (Duration, String, String);

fn read_wal_record<R: Read>(reader: &mut R) -> Result<WalRecord> {
    let secs = reader.read_u64::<Endian>()?;
    let nanos = reader.read_u32::<Endian>()?;
    let key = reader.read_varstring()?;
    let value = reader.read_varstring()?;
    Ok((Duration::new(secs, nanos), key, value))
}

pub struct WalWriter {
    dir: PathBuf,
    file: BufWriter<fs::File>,
}

impl WalWriter {
    pub fn new(path: &Path) -> Result<WalWriter> {
        let mut writer = BufWriter::new(fs::File::create(path.join(WAL_FILENAME))?);
        writer.write_all(WAL_SIGNATURE.as_bytes())?;
        writer.flush()?;
        Ok(WalWriter{
            dir: path.to_path_buf(),
            file: writer,
        })
    }

    // Only supports a single active WAL file for now, so no writing while
    // flushing the memtable sadly.
    pub fn reset(&mut self) -> Result<()> {
        self.file = BufWriter::new(fs::File::create(self.dir.join(WAL_FILENAME))?);
        self.file.write_all(WAL_SIGNATURE.as_bytes())?;
        self.file.flush()?;
        Ok(())
    }

    pub fn add(&mut self, stamp: &Duration, key: &str, value: &str) -> Result<()> {
        self.file.write_u64::<Endian>(stamp.as_secs())?;
        self.file.write_u32::<Endian>(stamp.subsec_nanos())?;
        self.file.write_varstring(key)?;
        self.file.write_varstring(value)?;
        self.file.flush()?;
        Ok(())
    }
}

pub struct WalReader {
    reader: Option<BufReader<fs::File>>,
    filesize: u64,
    bytesread: u64,
}

impl WalReader {
    pub fn new(dir: &Path) -> Result<Self> {
        let path = dir.join(WAL_FILENAME);
        if !path.exists() {
            // Just return no records if no WAL file exists
            return Ok(WalReader{reader: None, filesize: 0, bytesread: 0});
        }

        let mut f = BufReader::new(fs::File::open(path)?);
        let filesize = f.seek(SeekFrom::End(0))?;
        f.seek(SeekFrom::Start(0))?;

        let mut sig = [0; 4];
        f.read_exact(&mut sig)?;
        assert_eq!(sig, WAL_SIGNATURE.as_bytes());

        Ok(WalReader{reader: if filesize <= 4 { None } else { Some(f) },
                     filesize: filesize,
                     bytesread: 4})
    }
}


impl Iterator for WalReader {
    type Item = Result<(Duration, String, String)>;

    fn next(&mut self) -> Option<Self::Item> {

        let (dur, key, value) = match &mut self.reader {
            None => return None,
            Some(ref mut reader) => {
                let (dur, key, value) = match read_wal_record(reader) {
                    Ok(tuple) => tuple,
                    Err(err) => return Some(Err(err)),
                };

                self.bytesread += (8 + 4 + 4 + key.len() + 4 + value.len()) as u64;
                (dur, key, value)
            }
        };



        if self.bytesread == self.filesize {
            self.reader = None;
        }

        Some(Ok((dur, key, value)))
    }
}

#[cfg(test)]
mod test {
    use tempfile::{NamedTempFile, Builder as TempDirBuilder, TempDir};

    #[test]
    fn check_wal_works() {
        use crate::wal::{WalReader, WalWriter};
        use std::time::Duration;
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir().unwrap();
        {
            let mut writer = WalWriter::new(tmp_dir.path()).unwrap();

            writer.add(&Duration::new(10, 0), "foo", "x_foo").unwrap();
            writer.add(&Duration::new(20, 0), "bar", "x_bar").unwrap();
        }

        let mut reader = WalReader::new(tmp_dir.path()).unwrap();

        let (dur, key, val) = reader.next().unwrap().unwrap();
        assert_eq!(dur.as_secs(), 10);
        assert_eq!(key, "foo");
        assert_eq!(val, "x_foo");

        let (dur, key, val) = reader.next().unwrap().unwrap();
        assert_eq!(dur.as_secs(), 20);
        assert_eq!(key, "bar");
        assert_eq!(val, "x_bar");
    }
}
