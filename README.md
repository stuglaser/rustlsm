This is a toy implementation of a [log structured merge tree](https://en.wikipedia.org/wiki/Log-structured_merge-tree) in Rust, similar to [leveldb](http://leveldb.org/) or [rocksdb](https://rocksdb.org/).

This library is licensed under the MIT license.

It allows fast writes, performs compactions in the background to speed up reads as much as possible, and includes a write-ahead log so it doesn't lose data.  See `src/example.rs` for an example of usage.

### Example

```
use rustlsm::{LsmTree, Options};
fn ... {
    let mut tree = LsmTree::new(path, Options::default())?;
    tree.set("foo", "bar")?;
    // ...

    assert_eq!(tree.get("foo")?, Some("bar"));
}
```

_Written by Stu Glaser_
