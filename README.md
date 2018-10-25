This is a toy implementation of a [log structured merge tree](https://en.wikipedia.org/wiki/Log-structured_merge-tree) in Rust, similar to [leveldb](http://leveldb.org/) or [rocksdb](https://rocksdb.org/).

This library is licensed under the MIT license.

It allows fast writes, performs compactions to speed up reads as much as possible, and includes a write-ahead log so it doesn't lose data.  See `src/example.rs` for an example of usage.

_Written by Stu Glaser_
