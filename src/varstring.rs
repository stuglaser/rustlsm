use std::io;

extern crate byteorder;
use byteorder::*;
use byteorder::LittleEndian as Endian;

/// Helpers for IO on strings with lengths
pub trait VarStringIORead : io::Read {
    fn read_varstring(&mut self) -> Result<String, io::Error> {
        let len = self.read_u32::<Endian>()? as usize;

        let mut buf = vec![0 as u8; len];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf).unwrap())
    }
}

impl<R: io::Read + ?Sized> VarStringIORead for R {}

pub trait VarStringIOWrite : io::Write{
    fn write_varstring(&mut self, s: &str) -> Result<(), io::Error> {
        let bytes = s.as_bytes();
        self.write_u32::<Endian>(bytes.len() as u32)?;
        self.write_all(bytes)?;
        Ok(())
    }
}

impl<W: io::Write + ?Sized> VarStringIOWrite for W {}
