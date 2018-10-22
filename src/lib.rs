//#![allow(dead_code, unused)]

extern crate byteorder;
extern crate rand;
extern crate tempfile;
extern crate uuid;
extern crate owning_ref;

use std::cmp::max;
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::str;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use byteorder::*;
use byteorder::LittleEndian as Endian;

mod wal;
use wal::{WalReader, WalWriter};

mod varstring;
use varstring::{VarStringIORead, VarStringIOWrite};

pub mod sstable;
use sstable::{SSTable, SSTableBuilder, SSTableChainer};

pub struct Options {
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
        Options{sstable_size: 4096,
                level0_size: 4,
                level1_size: 10,
                level_size_factor: 10,
                start_compaction_thread: true}
    }

    // Options for testing; makes everything tiny so compactions are more
    // serious.
    pub fn tiny() -> Options {
        Options{sstable_size: 1,
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
        if self.current_builder.estimated_total_size() >= self.size_threshold {
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
    memtable_size: usize,  // Estimate of bytes of the memtable
    slabs: Vec<SlabInfo>,
}

enum Which {
    Finished,
    TargetNext,
    OverlapsNext,
    BothNext,
}

impl LsmTreeInner {
    fn new(path: &Path, options: Options) -> Result<Self, io::Error> {
        let mut tree = Self{ path: path.to_path_buf(),
                             options: options,
                             map: BTreeMap::new(),
                             memtable_size: 0,
                             slabs: Vec::new(), };
        tree.load_metadata()?;
        Ok(tree)
    }

    fn set(&mut self, key: &str, value: &str) {
        self.map.insert(key.to_string(), value.to_string());

        // Very rough estimate of memtable size, including the expected SSTable
        // index size.
        self.memtable_size += 2 * key.len() + value.len() + 4 + 4 + 8;
    }

    fn get(&self, key: &str) -> Result<Option<String>, io::Error> {
        if let Some(s) = self.map.get(key) {
            return Ok(Some(s.to_string()));
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
                let mut sstable = SSTable::open(&path)?;
                if let Some(v) = sstable.get(key)? {
                    if slab.level <= level {
                        level = slab.level;
                        value = v;
                    }
                }
            }
        }

        Ok(match level{
            999 => None,  // Not found
            _ => Some(value),
        })
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
        fs::rename(temppath, self.path.join("METADATA"))?;
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

    fn new_slab_filename(level: usize) -> String {
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
        let new_slab = SlabInfo{
            level: 0,
            key_min: self.map.keys().next().unwrap().to_string(),
            key_max: self.map.keys().next_back().unwrap().to_string(),
            filename: Self::new_slab_filename(0) };

        // Writes the slab to disk
        let mut builder = SSTableBuilder::create(&self.path.join(&new_slab.filename))?;
        for item in &self.map {
            builder.add(&item.0, &item.1)?;
        }
        builder.finish()?;

        self.slabs.push(new_slab);
        self.flush_metadata()?;
        self.map.clear();
        self.memtable_size = 0;

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
        let mut count = vec![0; 50]; // TODO: max levels assumed
        let mut max_level : usize = 0;
        for slab in &self.slabs {
            count[slab.level] += 1;
            max_level = max(max_level, slab.level);
        }

        // Finds the lowest level that's too big
        if count[0] > self.options.level0_size {
            return Some(0);
        }
        else {
            let mut max_size = self.options.level1_size;
            for i in 1..(max_level+1) {
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
    //
    // Needs to be a static function so we don't hold a lock on the tree while
    // we merge files.
    fn prepare_compaction(path: &Path, sstable_size: usize,
                          target: &SlabInfo, overlaps: &Vec<SlabInfo>) ->
        Result<Vec<SlabInfo>, io::Error>
    {
        // Sets up iteration over the target sstable to merge from
        let target_sstable = SSTable::open(&path.join(&target.filename))?;
        let mut iter_target = target_sstable.iter().peekable();

        // Sets up iteration over the overlaps
        let overlaps_paths = overlaps.iter().map(
            |info| path.join(&info.filename)).collect();
        let mut iter_overlaps = SSTableChainer::new(overlaps_paths)?.peekable();

        // Output streamer
        let mut streamer = SSTableStreamWriter::new(
            path,
            target.level + 1,
            || Self::new_slab_filename(target.level + 1),
            sstable_size * 1024)?;

        // Here we go.  We will...
        // .. merge iter_target and iter_overlaps
        // .. into streamer
        // .. doing bookkeeping in new_slabs

        loop {
            use ::Which::*;
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

pub struct LsmTree {
    tree: Arc<RwLock<LsmTreeInner>>,
    wal_writer: WalWriter,
}

impl LsmTree {
    pub fn new(path: &Path, options: Options) -> Result<Self, io::Error> {
        let should_compact = options.start_compaction_thread;
        let mut inner = LsmTreeInner::new(path, options)?;

        // Recovers data from the WAL. This is currently brittle; since the WAL
        // is only a single overwritten file, we must recover (and flush) before
        // creating the WalWriter below.
        for datum in WalReader::new(path)? {
            let (_stamp, key, value) = datum?;
            inner.set(&key, &value);
        }
        inner.flush()?;

        let tree = Arc::new(RwLock::new(inner));
        if should_compact {
            Self::start_compaction_thread(tree.clone());
        }
        Ok(Self{tree: tree, wal_writer: WalWriter::new(path)?})
    }

    fn nowstamp() -> Duration {
        use std::time::SystemTime;
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap()
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<(), io::Error>{
        let now = Self::nowstamp();
        let mut g = self.tree.write().unwrap();
        (*g).set(key, value);
        self.wal_writer.add(&now, key, value)?;

        if g.options.start_compaction_thread &&
            g.memtable_size >= g.options.sstable_size * 1024
        {
            g.flush()?;
            self.wal_writer.reset()?;
        }
        Ok(())
    }

    pub fn get(&self, key: &str) -> Result<Option<String>, io::Error> {
        let g = self.tree.read().unwrap();
        g.get(key)
    }

    /// Flushes the current memtable to a new slab in level0
    pub fn flush(&mut self) -> Result<(), io::Error> {
        let mut g = self.tree.write().unwrap();
        g.flush()?;
        self.wal_writer.reset()
    }

    /// Just a convenience function for testing. Typically you should let the
    /// internal thread handle compactions.
    pub fn maybe_compact(&mut self) -> Result<bool, io::Error> {
        Self::maybe_compact_internal(&self.tree)
    }

    // Needs to be a static function so it can be called from the compaction
    // thread.
    fn maybe_compact_internal(lock: &RwLock<LsmTreeInner>) -> Result<bool, io::Error> {
        let (path, sstable_size, target, overlaps) = {
            // Only need a read-lock to set up the compaction
            let tree = lock.read().unwrap();

            let level_to_compact = match tree.choose_level_to_compact() {
                None => return Ok(false),  // No compaction needed
                Some(level) => level,
            };

            println!("Going to compact level {}", level_to_compact);

            let (target, overlaps) = tree.select_compaction_targets(level_to_compact);
            (tree.path.clone(), tree.options.sstable_size, target, overlaps)
        };

        // Merging the sstables must happen while the tree is unlocked so other
        // operations can access the data.
        let new_slabs = LsmTreeInner::prepare_compaction(
            &path, sstable_size, &target, &overlaps)?;

        // Need a write-lock to commit the compaction
        println!("Locking to commit compaction");
        let mut tree = lock.write().unwrap();
        tree.commit_compaction(target, overlaps, new_slabs)?;
        println!("Compaction committed");

        Ok(true)
    }

    // What do we need to do?
    //
    // We want a reader lock on the slabs for `get`
    // Need a writer lock on the memtable for `put` and `compact0`
    // Need a writer lock on the slabs for part of `compact`
    fn start_compaction_thread(mutex: Arc<RwLock<LsmTreeInner>>) {
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
                    else { println!("long sleep"); 250 }
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
        let tree = LsmTree::new(tmp_dir.path(), Options::tiny())?;
        println!("Created temp dir {:?}", tmp_dir);
        return Result::Ok((tmp_dir, tree));
    }

    #[test]
    fn sanity_single_put_get() {
        let (_dir, mut tree) = create_temp_db().unwrap();
        tree.set("foo", "bar").unwrap();
        assert_eq!(tree.get("foo").unwrap(), Some("bar".to_string()));
    }

    #[test]
    fn check_persists_single_key_flush() {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir().unwrap();
        {
            let mut tree = LsmTree::new(tmp_dir.path(), Options::tiny()).unwrap();
            println!("Created temp dir {:?}", tmp_dir);
            tree.set("foo", "bar").unwrap();
            tree.flush().unwrap();
        }

        {
            let tree = LsmTree::new(tmp_dir.path(), Options::tiny()).unwrap();
            assert_eq!(tree.get("foo").unwrap(), Some("bar".to_string()));
        }
    }

    #[test]
    fn check_persists_single_key_noflush() {
        let tmp_dir = TempDirBuilder::new().prefix("rustlsm_test").tempdir().unwrap();
        {
            let mut tree = LsmTree::new(tmp_dir.path(), Options::tiny()).unwrap();
            println!("Created temp dir {:?}", tmp_dir);
            tree.set("foo", "bar").unwrap();
        }

        {
            let tree = LsmTree::new(tmp_dir.path(), Options::tiny()).unwrap();
            assert_eq!(tree.get("foo").unwrap(), Some("bar".to_string()));
        }
    }

    #[test]
    fn single_value_compact() -> Result<(), io::Error> {
        let (_dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "bar").unwrap();
        tree.flush().unwrap();
        tree.maybe_compact().unwrap();
        assert_eq!(tree.get("foo").unwrap(), Some("bar".to_string()));
        Ok(())
    }

    #[test]
    fn double_compact() {
        let (dir, mut tree) = create_temp_db().unwrap();

        tree.set("foo", "valfoo").unwrap();
        tree.flush().unwrap();
        tree.set("bar", "valbar").unwrap();

        assert_eq!(tree.get("foo").unwrap(), Some("valfoo".to_string()), "after 1 compact");
        assert_eq!(tree.get("bar").unwrap(), Some("valbar".to_string()), "after 1 compact");

        tree.flush().unwrap();
        assert_eq!(tree.get("foo").unwrap(), Some("valfoo".to_string()), "after 2 compacts");
        assert_eq!(tree.get("bar").unwrap(), Some("valbar".to_string()), "after 2 compacts");

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
    fn check_size() {
        let letters = "abcdefg";

        // Creates the SSTable
        let file = NamedTempFile::new().unwrap();

        let mut builder = SSTableBuilder::create(file.path()).unwrap();
        for ch in letters.chars() {
            let kv = format!("{}", ch);
            builder.add(&kv, &kv).unwrap();
        }
        builder.finish().unwrap();

        let mut sstable = SSTable::open(file.path()).unwrap();
        assert_eq!(sstable.len(), letters.len());
    }

    #[test]
    fn check_for_compactions() {
        let (_dir, mut tree) = create_temp_db().unwrap();

        // Random data
        let mut rng = make_seeded_rng::<IsaacRng>(53335);
        let keys = make_random_keys(&mut rng, 300);

        let mut num_compactions : usize = 0;
        for (i, key) in keys.iter().enumerate() {
            tree.set(key, &format!("x_{}", key)).unwrap();
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
            assert_eq!(tree.get(&key).unwrap(), Some(value));
        }

        assert!(num_compactions > 2);  // Checks that any compactions happened
    }
}
