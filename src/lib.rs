#![allow(dead_code, unused)]

extern crate byteorder;
extern crate rand;
extern crate tempfile;
extern crate uuid;
extern crate owning_ref;

use std::cmp::max;
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::io::{Read, Write, Seek, SeekFrom};
use std::io::ErrorKind::UnexpectedEof;
use std::path::Path;
use std::path::PathBuf;
use std::str;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use byteorder::*;
use byteorder::LittleEndian as Endian;
use owning_ref::OwningHandle;


const SSTABLE_SIGNATURE : &'static str = "SSTB";

/// Helpers for IO on strings with lengths
trait VarStringIO {
    fn write_varstring(&mut self, s: &str) -> Result<(), io::Error>;
    fn read_varstring(&mut self) -> Result<String, io::Error>;
}
impl VarStringIO for fs::File {
    fn write_varstring(&mut self, s: &str) -> Result<(), io::Error> {
        let bytes = s.as_bytes();
        self.write_u32::<Endian>(bytes.len() as u32)?;
        self.write_all(bytes)?;
        Ok(())
    }

    fn read_varstring(&mut self) -> Result<String, io::Error> {
        let len = self.read_u32::<Endian>()? as usize;

        let mut buf = vec![0 as u8; len];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8(buf).unwrap())
    }
}

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
                -> Result<usize, io::Error> {
    let keybytes = key.as_bytes();
    let valuebytes = value.as_bytes();
    file.write_u32::<Endian>(keybytes.len() as u32)?;
    file.write_u32::<Endian>(valuebytes.len() as u32)?;
    file.write_all(keybytes)?;
    file.write_all(valuebytes)?;
    Ok((4 + 4 + keybytes.len() + valuebytes.len()))
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
    pub fn open(path: &Path) -> Result<SSTable, io::Error> {
        let mut sstable = SSTable{ path: path.to_path_buf(),
                                   size: 0,
                                   index: HashMap::new() };
        sstable.preload()?;
        Ok(sstable)
    }

    fn preload(&mut self) -> Result<(), io::Error> {
        let mut file = fs::File::open(&self.path)?;

        // Reads the footer
        file.seek(SeekFrom::End(-20))?;
        let footer_loc = file.seek(SeekFrom::Current(0))?;

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
    bytes_written: usize,
    finished: bool,
}

impl SSTableBuilder {
    pub fn create(path: &Path) -> Result<SSTableBuilder, io::Error> {
        Ok(SSTableBuilder{
            file: fs::File::create(path)?,
            index: Vec::new(),
            bytes_written: 0,
            finished: false})
    }

    pub fn add(&mut self, key: &str, value: &str) -> Result<(), io::Error> {
        assert!(!self.finished);
        let loc = self.file.seek(SeekFrom::Current(0))?;
        self.index.push((key.to_string(), loc));
        let bytes = write_record(&mut self.file, key, value)?;
        self.bytes_written += bytes;
        Ok(())
    }

    pub fn bytes_written(&self) -> usize {
        self.bytes_written
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

        // Writes the footer
        self.file.write_all(SSTABLE_SIGNATURE.as_bytes())?;
        self.file.write_u64::<Endian>(self.index.len() as u64)?;
        self.file.write_u64::<Endian>(index_loc as u64)?;

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
            self.finish();
        }
    }
}

/// Holds an SSTable and an iter to it together
type SSTableIterHandle = OwningHandle<Box<SSTable>, Box<SSTableIter<'static>>>;

/// For iterating through a list of SSTable's
struct SSTableChainer {
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

pub struct Options {
    /// Max length of the memtable
    memtable_len: usize,

    /// Target size of an sstable in kb
    sstable_size: usize,

    /// Max sstables in level0 and level1
    level0_size: usize,
    level1_size: usize,
    level_size_factor: usize,

    start_compaction_thread: bool,
}

impl Options {
    pub fn default() -> Options {
        Options{memtable_len: 256,
                sstable_size: 4096,
                level0_size: 4,
                level1_size: 10,
                level_size_factor: 10,
                start_compaction_thread: true}
    }

    // Options for testing; makes everything tiny so compactions are more
    // serious.
    pub fn tiny() -> Options {
        Options{memtable_len: 8,  // Irrelevent since we don't compact automatically here
                sstable_size: 1,
                level0_size: 2,
                level1_size: 4,
                level_size_factor: 2,
                start_compaction_thread: false}
    }
}

/// Describes a particular SSTable
#[derive(Clone, Debug)]
struct SlabInfo {
    level: usize,
    key_min: String,
    key_max: String,
    filename: String,
}

impl SlabInfo {
    pub fn overlaps(&self, other : &SlabInfo) -> bool {
        !(self.key_max < other.key_min ||
          self.key_min > other.key_max)
    }
}

/// Writes out a sequence of SSTables with a size threshold
struct SSTableStreamWriter<Namer>
    where Namer: Fn() -> String
{
    dir: PathBuf,
    level: usize,
    new_file_fn: Namer,
    size_threshold: usize,

    current_filename: String,
    current_builder: SSTableBuilder,
    slabs: Vec<SlabInfo>,
}

impl<Namer> SSTableStreamWriter<Namer>
    where Namer: Fn() -> String
{
    pub fn new(dir: &Path,
               level: usize,
               new_file_fn: Namer,
               size_threshold: usize) -> Result<Self, io::Error>
    {
        let filename = new_file_fn();
        let builder = SSTableBuilder::create(&dir.join(&filename))?;
        Ok(Self{
            dir: dir.to_path_buf(),
            level: level,
            new_file_fn: new_file_fn,
            size_threshold: size_threshold,
            current_filename: filename,
            current_builder: builder,
            slabs: Vec::new(),
        })
    }

    fn finalize_current_sstable(&mut self) -> Result<(), io::Error> {
        {
            let key_bounds = self.current_builder.key_bounds();
            self.slabs.push(SlabInfo{
                level: self.level,
                key_min: key_bounds.0.to_string(),
                key_max: key_bounds.1.to_string(),
                filename: self.current_filename.clone(),
            });
        }
        self.current_builder.finish()
    }

    pub fn add(&mut self, key: &str, value: &str) -> Result<(), io::Error> {
        if self.current_builder.bytes_written() > self.size_threshold {
            self.finalize_current_sstable()?;

            // Creates a new SSTable
            self.current_filename = (self.new_file_fn)();
            self.current_builder = SSTableBuilder::create(
                &self.dir.join(&self.current_filename))?;
        }

        self.current_builder.add(key, value)
    }

    pub fn finish(mut self) -> Result<Vec<SlabInfo>, io::Error> {
        self.finalize_current_sstable()?;
        Ok(self.slabs)
    }
}

struct LsmTreeInner {
    path: PathBuf,
    options: Options,
    map: BTreeMap<String, String>,
    slabs: Vec<SlabInfo>,
}

enum Which {
    Finished,
    TargetNext,
    OverlapsNext,
    BothNext,
}

impl LsmTreeInner {
    fn new(path: &Path, options: Options) -> Self {
        let mut tree = Self{ path: path.to_path_buf(),
                             options: options,
                             map: BTreeMap::new(),
                             slabs: Vec::new(), };
        tree.load_metadata();
        tree
    }

    fn set(&mut self, key: &str, value: &str) {
        self.map.insert(key.to_string(), value.to_string());
    }

    fn get(&self, key: &str) -> Option<String> {
        if let Some(s) = self.map.get(key) {
            return Some(s.to_string());
        }

        // TODO: There is some magic baked in here. Slabs in level0 can have
        // overlapping key ranges, so if they disagree which value should we
        // use? Since we always append the latest slab to `slabs`, the correct
        // value to use is the one from the level0 slab that comes latest in the
        // `slabs` list.  If we change how slabs are stored, this will break.

        // Searches through the slabs
        let mut level : usize = 999;
        let mut value = String::new();
        for slab in &self.slabs {
            if slab.key_min.as_str() <= key && key <= slab.key_max.as_str() {
                let path = self.path.join(&slab.filename);
                let mut sstable = SSTable::open(&path).unwrap();  // TODO: poor unwrap
                if let Some(v) = sstable.get(key).unwrap() { // TODO: poor unwrap
                    if slab.level <= level {
                        level = slab.level;
                        value = v;
                    }
                }
            }
        }

        match level{
            999 => None,  // Not found
            _ => Some(value),
        }
    }

    fn flush_metadata(&mut self) -> Result<(), io::Error> {
        let temppath = self.path.join("METADATA.temp");
        let mut file = fs::File::create(&temppath)?;

        file.write_u64::<Endian>(self.slabs.len() as u64)?;
        for slab in &self.slabs {
            file.write_u16::<Endian>(slab.level as u16)?;
            file.write_varstring(&slab.key_min)?;
            file.write_varstring(&slab.key_max)?;
            file.write_varstring(&slab.filename)?;
        }

        file.sync_all()?;
        drop(file);  // Flush
        fs::rename(temppath, self.path.join("METADATA"));
        Ok(())
    }

    fn load_metadata(&mut self) -> Result<(), io::Error> {
        let path = self.path.join("METADATA");
        if !path.exists() {
            return Ok(())  // Nothing to load
        }
        let mut file = fs::File::open(path)?;

        let size = file.read_u64::<Endian>()?;
        let mut slabs : Vec<SlabInfo> = Vec::new();
        for _ in 0..size {
            let level = file.read_u16::<Endian>()? as usize;
            let key_min = file.read_varstring()?;
            let key_max = file.read_varstring()?;
            let filename = file.read_varstring()?;
            slabs.push(SlabInfo{ level: level,
                                 key_min: key_min, key_max: key_max,
                                 filename: filename });
        }

        self.slabs = slabs;
        Ok(())
    }

    fn new_slab_filename(&self, level: usize) -> String {
        use uuid::Uuid;
        let uuid = Uuid::new_v4();
        format!("slab_l{:02}_{}.sst", level, uuid.to_hyphenated())
    }

    /// Flushes the current memtable to a new slab in level0
    pub fn flush(&mut self) -> Result<(), io::Error> {
        if self.map.is_empty() {
            return Ok(())  // Nothing to do
        }

        // Specifies the new slab
        use uuid::Uuid;
        let uuid = Uuid::new_v4();
        let new_slab = SlabInfo{
            level: 0,
            key_min: self.map.keys().next().unwrap().to_string(),
            key_max: self.map.keys().next_back().unwrap().to_string(),
            filename: self.new_slab_filename(0) };

        // Writes the slab to disk
        let mut builder = SSTableBuilder::create(&self.path.join(&new_slab.filename))?;
        for item in &self.map {
            builder.add(&item.0, &item.1)?;
        }
        builder.finish()?;

        self.slabs.push(new_slab);
        self.flush_metadata()?;
        self.map.clear();

        Ok(())
    }

    /// Selects an SSTable, from a certain level, that should be compacted
    fn get_target_slab(&self, level: usize) -> SlabInfo {
        let level_sstables : Vec<&SlabInfo> = self.slabs.iter()
            .filter(|s| s.level == level as usize)
            .collect();
        assert!(level_sstables.len() > 0); // Sanity


        if level == 0 {
            // We absolutely must choose the oldest SSTable on the first
            // level to compact. This way older values go to lower
            // levels before newer values.
            return level_sstables[0].clone();
        }

        // Selects a random sstable to compact.
        // TODO: There are better algorithms
        use rand::Rng;
        let n : usize = rand::thread_rng().gen_range(0, level_sstables.len() - 1);
        let target = level_sstables[n];
        target.to_owned()
    }

    /// Returns a level which is overfull
    pub fn choose_level_to_compact(&self) -> Option<usize> {
        // Counts the number of sstables at each level
        let mut count = vec![0; 20]; // TODO: max levels assumed
        for slab in &self.slabs {
            count[slab.level] += 1;
        }

        // Finds the lowest level that's too big
        if count[0] > self.options.level0_size {
            return Some(0);
        }
        else {
            let mut max_size = self.options.level1_size;
            for i in 1..count.len() {
                if count[i] > max_size {
                    return Some(i);
                }
                max_size *= self.options.level_size_factor;
            }
        }

        None
    }

    /// Returns an SSTable at the current level and its overlaps that would make
    /// up a single compaction step.
    fn select_compaction_targets(&self, level_to_compact: usize) ->
        (SlabInfo, Vec<SlabInfo>)
    {
        // Picks a target SSTable to compact
        let target = self.get_target_slab(level_to_compact as usize);

        // Finds the SSTables to merge the target with
        let overlaps : Vec<SlabInfo> = {
            let mut overlaps : Vec<&SlabInfo> = self.slabs.iter()
                .filter(|s|
                        s.level == target.level + 1 &&
                        target.overlaps(s))
                .collect();
            overlaps.sort_unstable_by_key(|s| &s.key_min);
            let overlaps : Vec<SlabInfo> = overlaps.drain(..).map(|s| s.clone()).collect();
            overlaps
        };

        (target, overlaps)
    }

    /// Writes the files for the compaction.
    fn prepare_compaction(&self, target: &SlabInfo, overlaps: &Vec<SlabInfo>) ->
        Result<Vec<SlabInfo>, io::Error>
    {
        // Sets up iteration over the target sstable to merge from
        let target_sstable = SSTable::open(&self.path.join(&target.filename))?;
        let mut iter_target = target_sstable.iter().peekable();

        // Sets up iteration over the overlaps
        let overlaps_paths = overlaps.iter().map(
            |info| self.path.join(&info.filename)).collect();
        let mut iter_overlaps = SSTableChainer::new(overlaps_paths)?.peekable();

        // Output streamer
        let mut streamer = SSTableStreamWriter::new(
            &self.path,
            target.level + 1,
            || self.new_slab_filename(target.level + 1),
            self.options.sstable_size * 1024)?;

        // Here we go.  We will...
        // .. merge iter_target and iter_overlaps
        // .. into streamer
        // .. doing bookkeeping in new_slabs

        loop {
            use self::Which::*;
            let which = match (iter_target.peek(), iter_overlaps.peek()) {
                (None, None) => Finished,
                (Some(_), None) => TargetNext,
                (None, Some(_)) => OverlapsNext,
                (Some(Err(_)), _) => TargetNext, // Force error below
                (_, Some(Err(_))) => OverlapsNext, // Force error below
                (Some(Ok(target)), Some(Ok(overlap))) => {
                    if target.0 == overlap.0 { BothNext }
                    else if target.0 < overlap.0 { TargetNext }
                    else { OverlapsNext }
                },
            };

            match which {
                TargetNext => {
                    let record = iter_target.next().unwrap()?;
                    streamer.add(&record.0, &record.1)?;
                },
                OverlapsNext => {
                    let record = iter_overlaps.next().unwrap()?;
                    streamer.add(&record.0, &record.1)?;
                },
                BothNext => {
                    let record = iter_target.next().unwrap()?;
                    streamer.add(&record.0, &record.1)?;
                    iter_overlaps.next().unwrap()?;  // Drop
                },
                Finished => break
            }
        }
        let new_slabs = streamer.finish()?;
        Ok(new_slabs)
    }

    /// Commits the results of `prepare_compaction`
    fn commit_compaction(&mut self,
                         target: SlabInfo,
                         overlaps: Vec<SlabInfo>,
                         new_slabs: Vec<SlabInfo>) ->
        Result<(), io::Error>
    {
        let mut slabs_to_remove : HashSet<&str> = HashSet::new();
        slabs_to_remove.insert(&target.filename);
        for slab in &overlaps {
            slabs_to_remove.insert(&slab.filename);
        }

        // Rewrites the metadata
        self.slabs.retain(|s| !slabs_to_remove.contains(s.filename.as_str()));
        self.slabs.extend(new_slabs);
        self.flush_metadata()?;

        // Deletes old SSTables
        for filename in slabs_to_remove {
            let path = self.path.join(filename);
            assert!(path.exists());
            fs::remove_file(path)?;
        }
        Ok(())
    }

    pub fn dump_metadata(&self) {
        let mut tmp = self.slabs.clone();
        tmp.sort_unstable_by_key(|s| format!("{:02}__{}", s.level, s.key_min));
        for slab in tmp {
            println!("Slab: {:?}", slab);
        }
    }
}

struct LsmTree {
    tree: Arc<RwLock<LsmTreeInner>>,
    should_compact: bool,
}

impl LsmTree {
    pub fn new(path: &Path, options: Options) -> Self {
        let should_compact = options.start_compaction_thread;
        let tree = Arc::new(RwLock::new(LsmTreeInner::new(path, options)));
        if should_compact {
            Self::start_compaction_thread(tree.clone());
        }
        Self{tree: tree, should_compact: should_compact}
    }

    pub fn set(&mut self, key: &str, value: &str) {
        // TODO: `set` mostly just needs a write lock to the memtable
        let mut g = self.tree.write().unwrap();
        (*g).set(key, value);

        if g.options.start_compaction_thread && g.map.len() >= g.options.memtable_len {
            g.flush();
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        let g = self.tree.read().unwrap();
        g.get(key)
    }

    /// Flushes the current memtable to a new slab in level0
    pub fn flush(&mut self) -> Result<(), io::Error> {
        let mut g = self.tree.write().unwrap();
        g.flush()
    }

    /// Just a convenience function for testing. Typically you should let the
    /// internal thread handle compactions.
    pub fn maybe_compact(&mut self) -> Result<bool, io::Error> {
        Self::maybe_compact_internal(&self.tree)
    }

    // Needs to be a static function so it can be called from the compaction
    // thread.
    fn maybe_compact_internal(lock: &RwLock<LsmTreeInner>) -> Result<bool, io::Error> {
        let (target, overlaps, new_slabs) = {
            // Only need a read-lock to set up the compaction
            let tree = lock.read().unwrap();

            let level_to_compact = match tree.choose_level_to_compact() {
                None => return Ok(false),  // No compaction needed
                Some(level) => level,
            };

            let (target, overlaps) = tree.select_compaction_targets(level_to_compact);
            let new_slabs = tree.prepare_compaction(&target, &overlaps)?;

            (target, overlaps, new_slabs)
        };

        // Need a write-lock to commit the compaction
        let mut tree = lock.write().unwrap();
        tree.commit_compaction(target, overlaps, new_slabs)?;

        Ok(true)
    }

    // What do we need to do?
    //
    // We want a reader lock on the slabs for `get`
    // Need a writer lock on the memtable for `put` and `compact0`
    // Need a writer lock on the slabs for part of `compact`
    pub fn start_compaction_thread(mutex: Arc<RwLock<LsmTreeInner>>) {
        thread::spawn(move || {
            loop {
                let compacted = {
                    match Self::maybe_compact_internal(&mutex) {
                        Ok(c) => c,
                        Err(err) => {
                            // TODO: Store error so next API call receives it
                            panic!("Failure during compaction: {:?}", err);
                        }
                    }
                };

                thread::sleep(Duration::from_millis(
                    if compacted { 10 }
                    else { 250 }
                ));
            }
        });
    }

    pub fn dump_metadata(&self) {
        let g = self.tree.write().unwrap();
        g.dump_metadata()
    }
}

#[cfg(test)]
mod tests {
    use {LsmTree, Options, SSTable, SSTableBuilder};
    use tempfile::{NamedTempFile, Builder as TempDirBuilder, TempDir};
    use std::io;
    use rand::{Rng, SeedableRng, IsaacRng, self};

    fn create_temp_db() -> Result<(TempDir, LsmTree), io::Error> {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir()?;
        let tree = LsmTree::new(tmp_dir.path(), Options::tiny());
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
    fn check_persists_single_key_flush() {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir().unwrap();
        {
            let mut tree = LsmTree::new(tmp_dir.path(), Options::tiny());
            println!("Created temp dir {:?}", tmp_dir);
            tree.set("foo", "bar");
            tree.flush().unwrap();
        }

        {
            let tree = LsmTree::new(tmp_dir.path(), Options::tiny());
            assert_eq!(tree.get("foo"), Some("bar".to_string()));
        }
    }

    /*  TODO: add back in when we implement WAL
    #[test]
    fn check_persists_single_key_noflush() {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir().unwrap();
        {
            let mut tree = LsmTree::new(tmp_dir.path(), Options::tiny());
            println!("Created temp dir {:?}", tmp_dir);
            tree.set("foo", "bar");
        }

        {
            let tree = LsmTree::new(tmp_dir.path(), Options::tiny());
            assert_eq!(tree.get("foo"), Some("bar".to_string()));
        }
    }*/

    #[test]
    fn single_value_compact() -> Result<(), io::Error> {
        let (dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "bar");
        tree.flush().unwrap();
        tree.maybe_compact().unwrap();
        assert_eq!(tree.get("foo"), Some("bar".to_string()));
        Ok(())
    }

    #[test]
    fn double_compact() {
        let (dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "valfoo");
        tree.flush().unwrap();
        tree.set("bar", "valbar");

        assert_eq!(tree.get("foo"), Some("valfoo".to_string()), "after 1 compact");
        assert_eq!(tree.get("bar"), Some("valbar".to_string()), "after 1 compact");

        tree.flush().unwrap();
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
        builder.add("bar", "xbar");
        builder.add("foo", "xfoo");
        builder.finish();

        let mut sstable = SSTable::open(file.path()).unwrap();
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

        let mut sstable = SSTable::open(file.path()).unwrap();
        assert_eq!(sstable.len(), letters.len());
    }

    #[test]
    fn check_for_compactions() {
        let (dir, mut tree) = create_temp_db().unwrap();

        // Random data
        let mut rng = make_seeded_rng::<IsaacRng>(53335);
        let mut keys = make_random_keys(&mut rng, 300);

        let mut num_compactions : usize = 0;
        for (i, key) in keys.iter().enumerate() {
            tree.set(key, &format!("x_{}", key));
            if (i + 1) % 10 == 0 {
                tree.flush().unwrap();
                while tree.maybe_compact().unwrap() {
                    num_compactions += 1;
                }
            }
        }

        //println!("{} compactions", num_compactions);
        //tree.dump_metadata();
        //dir.into_path();

        for key in keys {
            let value = format!("x_{}", key);
            assert_eq!(tree.get(&key), Some(value));
        }

        assert!(num_compactions > 2);  // Checks that any compactions happened
    }
}
