use crate::tree::*;
use std::{
    io::{self, Read},
    ops::Range,
};

fn slice_parts(len: ByteNum, range: Range<ByteNum>) -> impl Iterator<Item = StreamItem> {
    struct State {
        len: ByteNum,
        start: ByteNum,
        end: ByteNum,
        res: Vec<StreamItem>,
    }
    impl State {

        fn blocks(&self) -> BlockNum {
            blocks(self.len, BlockLevel(0))
        }

        fn hashes(&self) -> NodeNum {
            num_hashes(self.blocks())
        }

        fn contains(&self, block: BlockNum) -> bool {
            self.start_block() <= block && block < self.end_block()
        }

        // block in which the start of the range lies
        fn start_block(&self) -> BlockNum {
            BlockNum(self.start.0 / 1024)
        }

        // block in which the end of the range lies
        fn end_block(&self) -> BlockNum {
            BlockNum((self.end.0 + 1023) / 1024)
        }

        fn traverse(&mut self, offset: NodeNum) {
            let position = ByteNum((offset.0 + 1) / 2 * 1024);
            if level(offset).0 > 0 {
                let (left, right) = if self.end <= position {
                    (true, false)
                } else if self.start >= position {
                    (false, true)
                } else {
                    (true, true)
                };
                self.res.push(StreamItem::Hashes { left, right });
                if left {
                    self.traverse(left_child(offset).unwrap());
                }
                if right {
                    self.traverse(right_descendant(offset, self.hashes()).unwrap());
                }
            } else {
                let start = position;
                let end = (start + 1024).min(self.len);
                self.res.push(StreamItem::Data { start, size: (end - start).to_usize() })
            }
        }
    }
    let mut state = State {
        len,
        start: range.start,
        end: range.end,
        res: Vec::new(),
    };
    state.traverse(root(blocks(len, BlockLevel(0))));
    state.res.into_iter()
}

pub struct BlockIter {
    /// total len of the data in bytes
    len: ByteNum,
    /// start offset of the range we are interested in
    start: ByteNum,
    /// end offset of the range we are interested in
    end: ByteNum,
    // current offset
    offset: NodeNum,
    // done?
    done: bool,
}

#[derive(Debug)]
pub enum StreamItem {
    /// you will get 2 hashes, so 64 bytes.
    /// at least one of them will be relevant for later.
    Hashes {
        /// the left hash will be relevant for later and needs to be pushed on the stack
        left: bool,
        /// the right hash will be relevant for later and needs to be pushed on the stack
        right: bool,
    },
    /// you will get data for this range.
    /// you will need to verify this data against the hashes on the stack.
    /// the data to be actually returned can be just a subset of this.
    Data {
        /// start of the range, this is a multiple of a chunk size
        start: ByteNum,
        /// size of the range, this is at most 2 chunks
        size: usize
    },
}

impl BlockIter {
    fn blocks(&self) -> BlockNum {
        blocks(self.len, BlockLevel(0))
    }

    fn hashes(&self) -> NodeNum {
        num_hashes(self.blocks())
    }

    fn contains(&self, block: BlockNum) -> bool {
        self.start_block() <= block && block < self.end_block()
    }

    // block in which the start of the range lies
    fn start_block(&self) -> BlockNum {
        BlockNum(self.start.0 / 1024)
    }

    // block in which the end of the range lies
    fn end_block(&self) -> BlockNum {
        BlockNum((self.end.0 + 1023) / 1024)
    }
}

impl Iterator for BlockIter {
    type Item = StreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let position = BlockNum((self.offset.0 + 1) / 2);
        if level(self.offset).0 > 0 {
            if self.end_block() <= position {
                self.offset = left_child(self.offset).unwrap();
                Some(StreamItem::Hashes {
                    left: true,
                    right: false,
                })
            } else if self.start_block() >= position {
                self.offset = right_descendant(self.offset, self.hashes()).unwrap();
                Some(StreamItem::Hashes {
                    left: false,
                    right: true,
                })
            } else {
                self.offset = left_child(self.offset).unwrap();
                Some(StreamItem::Hashes {
                    left: true,
                    right: true,
                })
            }
        } else {
            let mut size = 0;
            let mut block = index(self.offset);
            if is_left_sibling(self.offset) {
                if self.contains(index(self.offset)) {
                    size += 1;
                    self.offset = self.offset + 2;
                } else {
                    block = block + 1;
                }
            }
            if self.contains(index(self.offset)) {
                size += 1;
            }
            if !self.contains(index(self.offset) + 1) {
                self.done = true;
            }
            let mut o = self.offset;
            loop {
                o = parent(o);
                if o > self.offset {
                    self.offset = o;
                    break;
                }
            }
            match right_descendant(self.offset, self.hashes()) {
                Some(offset) => self.offset = offset,
                None => {
                    self.done = true;
                },
            }
            let start = ByteNum(block.0 * 1024);
            let end = ByteNum((block.0 + size as u64) * 1024).min(self.len);
            Some(StreamItem::Data { start, size: (end - start).to_usize() })
        }
    }
}

pub struct DecoderInner<R> {
    inner: R,
    // range of blocks for which we get data
    blocks: Range<BlockNum>,
    /// total len of the data
    len: ByteNum,
    // offset of the hash on top of the stack
    offset: NodeNum,
    // hash stack
    stack: Vec<blake3::Hash>,
    // buffer
    buf: Vec<u8>,
}

impl<R> DecoderInner<R> {
    pub fn for_range(
        inner: R,
        hash: blake3::Hash,
        len: ByteNum,
        range: Range<ByteNum>,
        block_level: BlockLevel,
    ) -> Self {
        let mut range = range;
        range.start = range.start.min(len);
        range.end = range.end.min(len);
        let blocks = block_range(range, block_level);
        Self::new(inner, hash, blocks, len)
    }

    pub fn new(inner: R, hash: blake3::Hash, blocks: Range<BlockNum>, len: ByteNum) -> Self {
        Self {
            inner,
            blocks,
            len,
            offset: Self::root_offset(len),
            stack: vec![hash],
            buf: vec![0u8; block_size(BlockLevel(0)).to_usize()],
        }
    }

    pub fn into_inner(self) -> R {
        self.inner
    }

    fn is_root(&self) -> bool {
        self.offset == Self::root_offset(self.len)
    }

    fn root_offset(len: ByteNum) -> NodeNum {
        root(blocks(len, BlockLevel(0)))
    }

    fn block_level(&self) -> BlockLevel {
        BlockLevel(0)
    }
}

impl<R: Read> DecoderInner<R> {
    fn blocks(&self) -> BlockNum {
        blocks(self.len, self.block_level())
    }

    fn hashes(&self) -> NodeNum {
        num_hashes(self.blocks())
    }

    fn advance(&mut self) -> io::Result<u64> {
        if self.stack.is_empty() {
            return Ok(0);
        }
        loop {
            let position = BlockNum((self.offset.0 + 1) / 2);
            if level(self.offset).0 > 0 {
                // read two hashes
                let mut lh = [0u8; 32];
                let mut rh = [0u8; 32];
                self.inner.read_exact(&mut lh)?;
                self.inner.read_exact(&mut rh)?;
                let lh = blake3::Hash::from(lh);
                let rh = blake3::Hash::from(rh);
                // check if they match
                let parent = blake3::guts::parent_cv(&lh, &rh, self.is_root());
                let expected = self.stack.pop().unwrap();
                if parent != expected {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "hash mismatch"));
                }
                println!(
                    "validated {} {} using {}",
                    hex::encode(lh.as_bytes()),
                    hex::encode(rh.as_bytes()),
                    hex::encode(expected.as_bytes())
                );
                if self.blocks.end <= position {
                    println!("go left");
                    // we only need to go to the left
                    self.stack.push(lh);
                    self.offset = left_child(self.offset).unwrap();
                } else if self.blocks.start >= position {
                    println!("go right");
                    // we only need to go to the right
                    self.stack.push(rh);
                    self.offset = right_descendant(self.offset, self.hashes()).unwrap();
                } else {
                    // we need to visit both, but first left
                    println!("both");
                    self.stack.push(rh);
                    self.stack.push(lh);
                    self.offset = left_child(self.offset).unwrap();
                }
                break Ok(64);
            } else {
                let mut read = 0;
                if is_left_sibling(self.offset) {
                    read += self.read_and_validate_leaf()?;
                    self.offset = self.offset + 2;
                }
                read += self.read_and_validate_leaf()?;
                let offset0 = self.offset;
                while parent(self.offset) < self.offset {
                    self.offset = parent(self.offset);
                    println!("{:?}", self.offset);
                }
                self.offset = parent(self.offset);
                println!("{:?}", self.offset);
                match right_descendant(self.offset, self.hashes()) {
                    Some(offset) => self.offset = offset,
                    None => break Ok(read as u64),
                }
                println!("{:?} => {:?}", offset0, self.offset);
                break Ok(read as u64);
            }
        }
    }

    fn read_and_validate_leaf(&mut self) -> io::Result<usize> {
        let block = index(self.offset);
        if !self.blocks.contains(&block) {
            return Ok(0);
        }
        let size = leaf_size(block, self.block_level(), self.len).to_usize();
        self.inner.read_exact(&mut self.buf[..size])?;
        let hash = hash_block(block, &self.buf[..size], self.block_level(), self.is_root());
        let expected = self.stack.pop().unwrap();
        if hash != expected {
            println!(
                "leaf {:?} failed to validate using {}",
                block,
                hex::encode(expected.as_bytes())
            );
            return Err(io::Error::new(io::ErrorKind::InvalidData, "hash mismatch"));
        } else {
            println!(
                "leaf {:?} validated using {}",
                block,
                hex::encode(expected.as_bytes())
            );
        }
        Ok(size)
    }
}

fn print_blake(data: &[u8]) {
    assert!(data.len() >= 8);
    println!(
        "{} {}",
        hex::encode(&data[0..8]),
        u64::from_le_bytes(data[0..8].try_into().unwrap())
    );
    data[8..].chunks(32).for_each(|chunk| {
        let is_data = chunk.iter().all(|&x| x == chunk[0]);
        println!("{} {}", if is_data { "D" } else { "H" }, hex::encode(chunk))
    });
}

// impl<R: Read> Read for DecoderInner<R> {
//     fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
//         todo!()
//     }
// }

// impl<R: AsyncRead> AsyncRead for DecoderInner<R> {
//     fn poll_read(
//         self: std::pin::Pin<&mut Self>,
//         cx: &mut std::task::Context<'_>,
//         buf: &mut [u8],
//     ) -> std::task::Poll<std::io::Result<usize>> {
//         todo!()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use bao::decode::SliceDecoder;
    use bao::encode::SliceExtractor;
    use futures::StreamExt;
    use core::slice;
    use proptest::prelude::*;
    use std::io::{Cursor, Read};

    pub fn print_block_seqence(
        len: ByteNum,
        range: Range<ByteNum>,
    ) {
        let mut range = range;
        // map the range to our real range
        range.start = range.start.min(len);
        range.end = range.end.min(len);
        let iter = BlockIter {
            len,
            start: range.start,
            end: range.end,
            offset: root(blocks(len, BlockLevel(0))),
            done: false,
        };
        for item in iter {
            println!("{:?}", item);
        }

        println!("****");
        for item in slice_parts(len, range) {
            println!("{:?}", item);
        }
    }

    fn create_test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i / 1024) as u8).collect()
    }

    fn encode_slice(data: &[u8], slice_start: u64, slice_len: u64) -> (blake3::Hash, Vec<u8>) {
        let (encoded, hash) = bao::encode::encode(data);
        let mut extractor = SliceExtractor::new(Cursor::new(&encoded), slice_start, slice_len);
        let mut slice = vec![];
        extractor.read_to_end(&mut slice).unwrap();
        (hash, slice)
    }

    fn test_decode_all_impl(len: ByteNum) {
        // we need to be at block level 0 to be compatible with bao
        let block_level = BlockLevel(0);
        // create a slice encoding the entire data - equivalent to the bao inline encoding
        let (hash, slice) = encode_slice(&create_test_data(len.to_usize()), 0, len.0);
        // need to skip the length prefix
        let content = &slice[8..];
        // create an inner decoder to decode the entire slice
        let mut decoder = DecoderInner::new(
            Cursor::new(&content),
            hash,
            block_range(ByteNum(0)..len, block_level),
            len,
        );
        // advance until there is nothing left
        while let Ok(n) = decoder.advance() {
            if n == 0 {
                break;
            }
        }
        // check that we have read the entire slice
        let cursor = decoder.into_inner();
        assert_eq!(cursor.position(), content.len() as u64);
    }

    fn test_decode_part_impl(len: ByteNum, slice_start: ByteNum, slice_len: ByteNum) {
        // we need to be at block level 0 to be compatible with bao
        let block_level = BlockLevel(0);
        // create a slice encoding the given range
        let (hash, slice) = encode_slice(
            &create_test_data(len.to_usize()),
            slice_start.0,
            slice_len.0,
        );
        // need to skip the length prefix
        let content = &slice[8..];
        print_blake(&slice);
        // create an inner decoder to decode the entire slice
        let mut decoder = DecoderInner::for_range(
            Cursor::new(&content),
            hash,
            len,
            slice_start..slice_start + slice_len,
            block_level,
        );
        // advance until there is nothing left
        loop {
            let n = decoder.advance().unwrap();
            if n == 0 {
                break;
            }
        }
        // check that we have read the entire slice
        let cursor = decoder.into_inner();
        assert_eq!(cursor.position(), content.len() as u64);
    }

    fn size_start_len() -> impl Strategy<Value = (ByteNum, ByteNum, ByteNum)> {
        (0u64..65536)
            .prop_flat_map(|size| {
                let start = 0u64..size;
                let len = 0u64..size;
                (Just(size), start, len)
            })
            .prop_map(|(size, start, len)| {
                let size = ByteNum(size);
                let start = ByteNum(start);
                let len = ByteNum(len);
                (size, start, len)
            })
    }

    proptest! {
        #[test]
        fn test_decode_all(size in 1usize..32768) {
            test_decode_all_impl(ByteNum(size as u64));
        }

        #[test]
        fn test_decode_part((size, start, len) in size_start_len()) {
            test_decode_part_impl(size, start, len);
        }
    }

    #[test]
    fn test_decode_all_1() {
        test_decode_all_impl(ByteNum(1234456));
    }

    #[test]
    fn test_decode_all_2() {
        test_decode_all_impl(ByteNum(0));
        test_decode_all_impl(ByteNum(1));
        test_decode_all_impl(ByteNum(1024));
        test_decode_all_impl(ByteNum(1025));
        test_decode_all_impl(ByteNum(2049));
    }

    #[test]
    fn test_decode_part_1() {
        test_decode_part_impl(ByteNum(2048), ByteNum(0), ByteNum(1024));
        test_decode_part_impl(ByteNum(2048), ByteNum(1024), ByteNum(1024));
        test_decode_part_impl(ByteNum(4096), ByteNum(0), ByteNum(1024));
        test_decode_part_impl(ByteNum(4096), ByteNum(1024), ByteNum(1024));
    }

    #[test]
    fn test_decode_part_2() {
        test_decode_part_impl(ByteNum(548), ByteNum(520), ByteNum(505));
    }

    #[test]
    fn test_decode_part_3() {
        test_decode_part_impl(ByteNum(2126), ByteNum(2048), ByteNum(1));
    }

    #[test]
    fn test_decode_part_4() {
        test_decode_part_impl(ByteNum(3073), ByteNum(1024), ByteNum(1025));
    }

    #[test]
    fn print_block_seqence_1() {
        print_block_seqence(ByteNum(32768 * 2 - 1), ByteNum(43453)..ByteNum(434530000));
    }

    #[test]
    fn stream_test() {
        use futures::Stream;
        use async_stream::stream;
        fn zero_to_three() -> impl Stream<Item = u32> {
            stream! {
                for i in 0..3 {
                    yield i;
                }
            }
        }

        let s = zero_to_three();
        tokio::pin!(s);
        for i in StreamIter(s) {
            println!("{}", i);
        }

        struct StreamIter<S>(S);

        impl<S: Stream + Unpin> Iterator for StreamIter<S> {
            type Item = S::Item;

            fn next(&mut self) -> Option<Self::Item> {
                let waker = futures::task::noop_waker_ref();
                let mut cx = std::task::Context::from_waker(waker);
                match self.0.poll_next_unpin(&mut cx) {
                    std::task::Poll::Ready(item) => item,
                    std::task::Poll::Pending => None,
                }
            }
        }
    }
}
