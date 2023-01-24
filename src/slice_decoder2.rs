use crate::tree::*;
use std::{
    io::{self, Read},
    ops::Range,
};

struct SliceIter {
    len: ByteNum,
    range: Range<ByteNum>,
    res: std::iter::Peekable<std::vec::IntoIter<StreamItem>>,
}

impl SliceIter {
    fn len(&self) -> ByteNum {
        self.len
    }

    fn range(&self) -> Range<ByteNum> {
        self.range.clone()
    }

    fn new(len: ByteNum, range: Range<ByteNum>) -> SliceIter {
        struct State {
            len: ByteNum,
            range: Range<ByteNum>,
            res: Vec<StreamItem>,
        }
        // make sure the range is within 0..len
        let mut range = range;
        range.start = range.start.min(len);
        range.end = range.end.min(len);
        impl State {
            fn hashes(&self) -> NodeNum {
                num_hashes(blocks(self.len, BlockLevel(0)))
            }

            fn traverse(&mut self, offset: NodeNum) {
                let position = ByteNum((offset.0 + 1) / 2 * 1024);
                let is_root = offset == root(blocks(self.len, BlockLevel(0)));
                if level(offset).0 > 0 {
                    let (left, right) = if self.range.end <= position {
                        (true, false)
                    } else if self.range.start >= position {
                        (false, true)
                    } else {
                        (true, true)
                    };
                    self.res.push(StreamItem::Hashes {
                        left,
                        right,
                        is_root,
                    });
                    if left {
                        self.traverse(left_child(offset).unwrap());
                    }
                    if right {
                        self.traverse(right_descendant(offset, self.hashes()).unwrap());
                    }
                } else {
                    let start = position;
                    let end = (start + 1024).min(self.len);
                    self.res.push(StreamItem::Data {
                        start,
                        size: (end - start).to_usize(),
                        is_root,
                    })
                }
            }
        }
        let mut state = State {
            len,
            range,
            res: vec![StreamItem::Header],
        };
        state.traverse(root(blocks(len, BlockLevel(0))));
        // todo: it is easy to make this a proper iterator, and even possible
        // to make it an iterator without any state, but let's just keep it simple
        // for now.
        Self {
            len: state.len,
            range: state.range,
            res: state.res.into_iter().peekable(),
        }
    }

    pub fn peek(&mut self) -> Option<&StreamItem> {
        self.res.peek()
    }

    /// prints a bao encoded slice
    ///
    /// this is a simple use case for how to use the slice iterator to figure
    /// out what is what.
    fn print_bao_encoded(len: ByteNum, range: Range<ByteNum>, slice: &[u8]) {
        let mut offset = 0;
        for item in SliceIter::new(len, range) {
            if slice.len() < offset + item.size() {
                println!("incomplete slice");
                return;
            }
            match item {
                StreamItem::Header => {
                    let data = &slice[offset..offset + 8];
                    println!(
                        "header  {} {}",
                        hex::encode(data),
                        u64::from_le_bytes(data.try_into().unwrap())
                    );
                }
                StreamItem::Hashes {
                    left,
                    right,
                    is_root,
                } => {
                    let data = &slice[offset..offset + 64];
                    let used = |b| if b { "*" } else { " " };
                    println!("hashes root={}", is_root);
                    println!("{} {}", hex::encode(&data[..32]), used(left));
                    println!("{} {}", hex::encode(&data[32..]), used(right));
                }
                StreamItem::Data {
                    start,
                    size,
                    is_root,
                } => {
                    let data = &slice[offset..offset + size];
                    println!(
                        "data range={}..{} root={}",
                        start.to_usize(),
                        start.to_usize() + size,
                        is_root
                    );
                    for chunk in data.chunks(32) {
                        println!("{}", hex::encode(chunk));
                    }
                }
            }
            println!("");
            offset += item.size();
        }
    }
}

impl Iterator for SliceIter {
    type Item = StreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.res.next()
    }
}

#[derive(Debug)]
pub enum StreamItem {
    /// expect a 8 byte header
    Header,
    /// you will get 2 hashes, so 64 bytes.
    /// at least one of them will be relevant for later.
    Hashes {
        /// the left hash will be relevant for later and needs to be pushed on the stack
        left: bool,
        /// the right hash will be relevant for later and needs to be pushed on the stack
        right: bool,
        /// is this branch the root
        is_root: bool,
    },
    /// you will get data for this range.
    /// you will need to verify this data against the hashes on the stack.
    /// the data to be actually returned can be just a subset of this.
    Data {
        /// start of the range, this is a multiple of a chunk size
        start: ByteNum,
        /// size of the range, this is at most 2 chunks
        size: usize,
        /// is this leaf the root
        is_root: bool,
    },
}

impl StreamItem {
    pub fn size(&self) -> usize {
        match self {
            StreamItem::Header => 8,
            StreamItem::Hashes { .. } => 64,
            StreamItem::Data { size, .. } => *size,
        }
    }
}

pub struct SliceValidator<R> {
    /// the inner reader
    inner: R,
    /// The slice iterator
    ///
    /// This is used to figure out what to expect from the reader.
    /// It also provides info like the total length and the range.
    iter: Result<SliceIter, Range<ByteNum>>,
    // hash stack for validation
    stack: Vec<blake3::Hash>,
    // buffer for incomplete items
    buf: [u8; 1024],
}

impl<R> SliceValidator<R> {
    fn new(inner: R, hash: blake3::Hash, start: u64, len: u64) -> Self {
        let range = start..start.saturating_add(len);
        Self {
            inner,
            iter: Err(ByteNum(range.start)..ByteNum(range.end)),
            stack: vec![hash],
            buf: [0; 1024],
        }
    }

    fn into_inner(self) -> R {
        self.inner
    }
}

impl<R: Read> SliceValidator<R> {
    fn next0(&mut self) -> io::Result<Option<StreamItem>> {
        // deal with the case where we don't yet have the header
        let iter = match &mut self.iter {
            Ok(iter) => iter,
            Err(range) => {
                // read the len
                self.inner.read_exact(&mut self.buf[0..8])?;
                let len = u64::from_le_bytes(self.buf[0..8].try_into().unwrap());
                // switch to the mode where we have the len
                let iter = SliceIter::new(ByteNum(len), range.clone());
                // store and return the iter
                self.iter = Ok(iter);
                // return the header
                return Ok(self.iter.as_mut().unwrap().next());
            }
        };

        // get next item - if there is none, we are done
        let item = match iter.peek() {
            Some(item) => item,
            None => return Ok(None),
        };

        if self.stack.is_empty() {
            return Ok(None);
        }

        // read the item, whatever it is. at this point it is either a hash or data
        self.inner.read_exact(&mut self.buf[0..item.size()])?;
        match item {
            StreamItem::Hashes {
                left,
                right,
                is_root,
            } => {
                let lc = blake3::Hash::from(<[u8; 32]>::try_from(&self.buf[0..32]).unwrap());
                let rc = blake3::Hash::from(<[u8; 32]>::try_from(&self.buf[32..64]).unwrap());
                let expected = self.stack.pop().unwrap();
                let actual = blake3::guts::parent_cv(&lc, &rc, *is_root);
                if expected != actual {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid branch hash",
                    ));
                }
                // push the hashes on the stack in *reverse* order
                if *right {
                    self.stack.push(rc);
                }
                if *left {
                    self.stack.push(lc);
                }
                Ok(iter.next())
            }
            StreamItem::Data {
                start,
                size,
                is_root,
            } => {
                debug_assert!(start.0 % 1024 == 0);
                let chunk = start.0 / 1024;
                let mut hasher = blake3::guts::ChunkState::new(chunk);
                hasher.update(&self.buf[0..*size]);
                let expected = self.stack.pop().unwrap();
                let actual = hasher.finalize(*is_root);
                if expected != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "invalid leaf hash",
                    ));
                }
                Ok(iter.next())
            }
            StreamItem::Header => {
                unreachable!("we already handled the header")
            }
        }
    }
}

impl<R: Read> Iterator for SliceValidator<R> {
    type Item = io::Result<StreamItem>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next0().transpose()
    }
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
        let is_root = self.offset == root(self.blocks());
        if level(self.offset).0 > 0 {
            if self.end_block() <= position {
                self.offset = left_child(self.offset).unwrap();
                Some(StreamItem::Hashes {
                    left: true,
                    right: false,
                    is_root,
                })
            } else if self.start_block() >= position {
                self.offset = right_descendant(self.offset, self.hashes()).unwrap();
                Some(StreamItem::Hashes {
                    left: false,
                    right: true,
                    is_root,
                })
            } else {
                self.offset = left_child(self.offset).unwrap();
                Some(StreamItem::Hashes {
                    left: true,
                    right: true,
                    is_root,
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
                }
            }
            let start = ByteNum(block.0 * 1024);
            let end = ByteNum((block.0 + size as u64) * 1024).min(self.len);
            Some(StreamItem::Data {
                start,
                size: (end - start).to_usize(),
                is_root,
            })
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
    use proptest::prelude::*;
    use std::io::{Cursor, Read};

    pub fn print_block_seqence(len: ByteNum, range: Range<ByteNum>) {
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
        for item in SliceIter::new(len, range) {
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

    fn test_decode_all_impl_2(len: ByteNum) {
        // create a slice encoding the entire data - equivalent to the bao inline encoding
        let (hash, slice) = encode_slice(&create_test_data(len.to_usize()), 0, len.0);
        let mut read = Cursor::new(&slice);
        let validator = SliceValidator::new(&mut read, hash, 0, len.0);
        for item in validator {
            assert!(item.is_ok());
        }
        // check that we have read the entire slice
        assert_eq!(read.position(), slice.len() as u64);
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
        SliceIter::print_bao_encoded(len, slice_start..slice_start + slice_len, &slice);
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

    fn test_decode_part_impl_2(len: ByteNum, slice_start: ByteNum, slice_len: ByteNum) {
        // create a slice encoding the given range
        let (hash, slice) = encode_slice(
            &create_test_data(len.to_usize()),
            slice_start.0,
            slice_len.0,
        );
        // SliceIter::print_bao_encoded(len, slice_start..slice_start + slice_len, &slice);
        // create an inner decoder to decode the entire slice
        let mut reader = Cursor::new(&slice);
        let validator = SliceValidator::new(&mut reader, hash, slice_start.0, slice_len.0);
        for item in validator {
            assert!(item.is_ok());
        }
        assert_eq!(reader.position(), slice.len() as u64);
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
            let len = ByteNum(size as u64);
            test_decode_all_impl(len);
            test_decode_all_impl_2(len);
        }

        #[test]
        fn test_decode_part((size, start, len) in size_start_len()) {
            test_decode_part_impl(size, start, len);
            test_decode_part_impl_2(size, start, len);
        }
    }

    #[test]
    fn test_decode_all_1() {
        let len = ByteNum(12343465);
        test_decode_all_impl(len);
        test_decode_all_impl_2(len);
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
}
