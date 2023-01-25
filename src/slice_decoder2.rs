use futures::{ready, AsyncRead, Stream, StreamExt};

use crate::tree::*;
use std::{
    io::{self, Read},
    ops::Range,
    pin::Pin,
    task::{Context, Poll},
};

pub struct SliceIter {
    len: u64,
    range: Range<u64>,
    res: Option<std::iter::Peekable<std::vec::IntoIter<StreamItem>>>,
}

impl SliceIter {
    pub fn new(len: u64, range: Range<u64>) -> Self {
        SliceIter {
            len,
            range,
            res: None,
        }
    }

    /// set the length of the slice
    ///
    /// this can only be done before the first call to next
    pub fn set_len(&mut self, len: u64) {
        assert!(self.res.is_none());
        self.len = len;
    }

    // todo: it is easy to make this a proper iterator, and even possible
    // to make it an iterator without any state, but let's just keep it simple
    // for now.
    fn iterate(len: u64, range: Range<u64>) -> Vec<StreamItem> {
        struct State {
            len: u64,
            range: Range<u64>,
            res: Vec<StreamItem>,
        }
        // make sure the range is within 0..len
        let mut range = range;
        range.start = range.start.min(len);
        range.end = range.end.min(len);
        impl State {
            fn hashes(&self) -> NodeNum {
                num_hashes(blocks(ByteNum(self.len), BlockLevel(0)))
            }

            fn traverse(&mut self, offset: NodeNum) {
                let position = (offset.0 + 1) / 2 * 1024;
                let is_root = offset == root(blocks(ByteNum(self.len), BlockLevel(0)));
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
                        end,
                        is_root,
                    });
                }
            }
        }
        let mut state = State {
            len,
            range,
            res: vec![],
        };
        state.traverse(root(blocks(ByteNum(len), BlockLevel(0))));
        state.res
    }

    pub fn peek(&mut self) -> Option<StreamItem> {
        match self.res {
            Some(ref mut res) => res.peek().cloned(),
            None => Some(StreamItem::Header),
        }
    }

    /// prints a bao encoded slice
    ///
    /// this is a simple use case for how to use the slice iterator to figure
    /// out what is what.
    pub fn print_bao_encoded(len: u64, range: Range<u64>, slice: &[u8]) {
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
                    end,
                    is_root,
                } => {
                    let size = end - start;
                    let data = &slice[offset..offset + size as usize];
                    println!("data range={}..{} root={}", start, start + size, is_root);
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

    fn next(&mut self) -> Option<StreamItem> {
        match self.res {
            Some(ref mut res) => res.next(),
            None => {
                self.res = Some(
                    SliceIter::iterate(self.len, self.range.clone())
                        .into_iter()
                        .peekable(),
                );
                Some(StreamItem::Header)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
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
        /// start of the range
        start: u64,
        /// end of the range
        end: u64,
        /// is this leaf the root
        is_root: bool,
    },
}

impl StreamItem {
    pub fn size(&self) -> usize {
        match self {
            StreamItem::Header => 8,
            StreamItem::Hashes { .. } => 64,
            StreamItem::Data { start, end, .. } => (end - start) as usize,
        }
    }

    pub fn is_data(&self) -> bool {
        matches!(self, StreamItem::Data { .. })
    }
}

pub struct SliceValidator<R> {
    /// the inner reader
    inner: R,

    /// The slice iterator
    ///
    /// This is used to figure out what to expect from the reader.
    ///
    /// It also stores and provides the total length and the slice range.
    iter: SliceIter,

    // hash stack for validation
    //
    // gets initialized with the root hash
    stack: Vec<blake3::Hash>,

    // buffer for incomplete items
    //
    // this is used for both reading and writing
    //
    // it can contain an 8 byte header, 64 bytes of hashes or up to 1024 bytes of data
    buf: [u8; 1024],

    // start of the buffer
    //
    // when incrementally reading a buffer in the async reader, this indicates the
    // start of the free part of the buffer
    //
    // when incrementally writing a buffer in both the sync and async reader, this
    // indicates the start of the occupied part of the buffer
    //
    // the overall length of the buffer is in both cases determined by the current item
    buf_start: usize,
}

impl<R> SliceValidator<R> {
    /// create a new slice validator for the given hash and range
    pub fn new(inner: R, hash: blake3::Hash, start: u64, len: u64) -> Self {
        let range = start..start.saturating_add(len);
        Self {
            inner,
            iter: SliceIter::new(0, range.start..range.end),
            stack: vec![hash],
            buf: [0; 1024],
            buf_start: 0,
        }
    }

    /// get back the wrapped reader
    pub fn into_inner(self) -> R {
        self.inner
    }

    fn range(&self) -> &Range<u64> {
        &self.iter.range
    }

    /// given a stream item, get the part of the buffer that is relevant for it
    fn get_buffer(&self, item: &StreamItem) -> &[u8] {
        match item {
            StreamItem::Header => &self.buf[0..8],
            StreamItem::Hashes { .. } => &self.buf[0..64],
            StreamItem::Data { start, end, .. } => {
                let range = self.range();
                let start1 = start.max(&range.start);
                let end1 = end.min(&range.end);
                let start2 = (start1 - start) as usize;
                let end2 = (end1 - start) as usize;
                &self.buf[start2..end2]
            }
        }
    }

    fn next_with_full_buffer(&mut self, item: StreamItem) -> Result<Option<StreamItem>, &str> {
        match item {
            StreamItem::Header => {
                let len = u64::from_le_bytes(self.buf[0..8].try_into().unwrap());
                self.iter.set_len(len);
            }
            StreamItem::Hashes {
                left,
                right,
                is_root,
            } => {
                let lc = blake3::Hash::from(<[u8; 32]>::try_from(&self.buf[0..32]).unwrap());
                let rc = blake3::Hash::from(<[u8; 32]>::try_from(&self.buf[32..64]).unwrap());
                let expected = self.stack.pop().unwrap();
                let actual = blake3::guts::parent_cv(&lc, &rc, is_root);
                if expected != actual {
                    return Err("invalid branch hash");
                }
                // push the hashes on the stack in *reverse* order
                if right {
                    self.stack.push(rc);
                }
                if left {
                    self.stack.push(lc);
                }
            }
            StreamItem::Data {
                start,
                end,
                is_root,
            } => {
                debug_assert!(start % 1024 == 0);
                let chunk = start / 1024;
                let mut hasher = blake3::guts::ChunkState::new(chunk);
                let size = end - start;
                hasher.update(&self.buf[0..size as usize]);
                let expected = self.stack.pop().unwrap();
                let actual = hasher.finalize(is_root);
                if expected != actual {
                    return Err("invalid leaf hash");
                }
            }
        }
        Ok(self.iter.next())
    }

    /// get the stream item we would get next
    fn peek(&mut self) -> Option<StreamItem> {
        self.iter.peek()
    }
}

impl<R: Read> Iterator for SliceValidator<R> {
    type Item = std::io::Result<StreamItem>;

    fn next(&mut self) -> Option<Self::Item> {
        let iter = &mut self.iter;

        // get next item - if there is none, we are done
        let item = iter.peek()?;

        // read the item, whatever it is
        if let Err(cause) = self.inner.read_exact(&mut self.buf[0..item.size()]) {
            return Some(Err(cause));
        }

        // validate and return the item
        self.next_with_full_buffer(item)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            .transpose()
    }
}

impl<R: AsyncRead + Unpin> SliceValidator<R> {
    /// fill the buffer with at least `size` bytes
    fn fill_buffer(&mut self, cx: &mut Context<'_>, size: usize) -> Poll<io::Result<()>> {
        debug_assert!(size <= self.buf.len());
        debug_assert!(self.buf_start <= size);
        while self.buf_start < size {
            let n = ready!(
                Pin::new(&mut self.inner).poll_read(cx, &mut self.buf[self.buf_start..size])?
            );
            if n == 0 {
                return Poll::Ready(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF",
                )));
            }
            self.buf_start += n;
        }
        self.buf_start = 0;
        Poll::Ready(Ok(()))
    }
}

impl<R: AsyncRead + Unpin> Stream for SliceValidator<R> {
    type Item = tokio::io::Result<StreamItem>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let current = match self.peek() {
            Some(item) => item,
            None => return Poll::Ready(None),
        };
        let size = current.size();
        ready!(self.fill_buffer(cx, size))?;

        let item = self
            .next_with_full_buffer(current)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e));

        Poll::Ready(item.transpose())
    }
}

pub struct SliceReader<R> {
    inner: SliceValidator<R>,
    current_item: Option<StreamItem>,
}

impl<R> SliceReader<R> {
    pub fn new(inner: R, hash: blake3::Hash, start: u64, len: u64) -> Self {
        Self {
            inner: SliceValidator::new(inner, hash, start, len),
            current_item: None,
        }
    }

    pub fn into_inner(self) -> R {
        self.inner.into_inner()
    }
}

impl<R: Read> Read for SliceReader<R> {
    fn read(&mut self, tgt: &mut [u8]) -> io::Result<usize> {
        loop {
            // if we have no current item, get the next one
            if self.current_item.is_none() {
                self.current_item = match self.inner.next().transpose()? {
                    Some(item) if item.is_data() => Some(item),
                    Some(_) => continue,
                    None => break Ok(0),
                };
                self.inner.buf_start = 0;
            }

            // if we get here we we have a data item.
            let item = self.current_item.as_ref().unwrap();
            let src = self.inner.get_buffer(item);
            if src.is_empty() {
                self.current_item = None;
                self.inner.buf_start = 0;
                continue;
            }
            let n = (src.len() - self.inner.buf_start).min(tgt.len());
            let end = self.inner.buf_start + n;
            tgt[0..n].copy_from_slice(&src[self.inner.buf_start..end]);
            if end < src.len() {
                self.inner.buf_start = end;
            } else {
                self.current_item = None;
                self.inner.buf_start = 0;
            }
            debug_assert!(n > 0, "we should have read something");
            break Ok(n);
        }
    }
}

impl<R: AsyncRead + Unpin> AsyncRead for SliceReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        tgt: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(loop {
            // if we have no current item, get the next one
            if self.current_item.is_none() {
                self.current_item = match ready!(self.inner.poll_next_unpin(cx)).transpose()? {
                    Some(item) if item.is_data() => Some(item),
                    Some(_) => continue,
                    None => break Ok(0),
                };
                self.inner.buf_start = 0;
            }

            // if we get here we we have a data item.
            let item = self.current_item.as_ref().unwrap();
            let src = self.inner.get_buffer(item);
            if src.is_empty() {
                self.current_item = None;
                self.inner.buf_start = 0;
                continue;
            }
            let n = (src.len() - self.inner.buf_start).min(tgt.len());
            let end = self.inner.buf_start + n;
            tgt[0..n].copy_from_slice(&src[self.inner.buf_start..end]);
            if end < src.len() {
                self.inner.buf_start = end;
            } else {
                self.current_item = None;
                self.inner.buf_start = 0;
            }
            debug_assert!(n > 0, "we should have read something");
            break Ok(n);
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bao::encode::SliceExtractor;
    use futures::AsyncReadExt;
    use proptest::prelude::*;
    use std::io::{Cursor, Read};

    fn create_test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i / 1024) as u8).collect()
    }

    /// Encode a slice of the given data and return the hash and the encoded slice, using the bao encoder
    fn encode_slice(data: &[u8], slice_start: u64, slice_len: u64) -> (blake3::Hash, Vec<u8>) {
        let (encoded, hash) = bao::encode::encode(data);
        let mut extractor = SliceExtractor::new(Cursor::new(&encoded), slice_start, slice_len);
        let mut slice = vec![];
        extractor.read_to_end(&mut slice).unwrap();
        (hash, slice)
    }

    /// Test implementation for the test_decode_all test, to be called by both proptest and hardcoded tests
    fn test_decode_all_impl(len: ByteNum) {
        // create a slice encoding the entire data - equivalent to the bao inline encoding
        let test_data = create_test_data(len.to_usize());
        let (hash, slice) = encode_slice(&test_data, 0, len.0);

        // test just validation without reading
        let mut cursor = Cursor::new(&slice);
        let validator = SliceValidator::new(&mut cursor, hash, 0, len.0);
        for item in validator {
            assert!(item.is_ok());
        }
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);

        // test validation and reading
        let mut cursor = Cursor::new(&slice);
        let mut reader = SliceReader::new(&mut cursor, hash, 0, len.0);
        let mut data = vec![];
        reader.read_to_end(&mut data).unwrap();
        assert_eq!(data, test_data);

        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);
    }

    /// Test implementation for the test_decode_all test, to be called by both proptest and hardcoded tests
    async fn test_decode_all_async_impl(len: ByteNum) {
        // create a slice encoding the entire data - equivalent to the bao inline encoding
        let test_data = create_test_data(len.to_usize());
        let (hash, slice) = encode_slice(&test_data, 0, len.0);

        // test just validation without reading
        let mut cursor = futures::io::Cursor::new(&slice);
        let mut validator = SliceValidator::new(&mut cursor, hash, 0, len.0);
        while let Some(item) = validator.next().await {
            assert!(item.is_ok());
        }
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);

        // test validation and reading
        let mut cursor = futures::io::Cursor::new(&slice);
        let mut reader = SliceReader::new(&mut cursor, hash, 0, len.0);
        let mut data = vec![];
        reader.read_to_end(&mut data).await.unwrap();
        assert_eq!(data, test_data);

        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);
    }

    /// Test implementation for the test_decode_part test, to be called by both proptest and hardcoded tests
    fn test_decode_part_impl(len: ByteNum, slice_start: ByteNum, slice_len: ByteNum) {
        let test_data = create_test_data(len.to_usize());
        // create a slice encoding the given range
        let (hash, slice) = encode_slice(&test_data, slice_start.0, slice_len.0);
        // SliceIter::print_bao_encoded(len, slice_start..slice_start + slice_len, &slice);

        // create an inner decoder to decode the entire slice
        let mut cursor = Cursor::new(&slice);
        let validator = SliceValidator::new(&mut cursor, hash, slice_start.0, slice_len.0);
        for item in validator {
            assert!(item.is_ok());
        }
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);

        let mut cursor = Cursor::new(&slice);
        let mut reader = SliceReader::new(&mut cursor, hash, slice_start.0, slice_len.0);
        let mut data = vec![];
        reader.read_to_end(&mut data).unwrap();
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);
        // check that we have read the correct data
        let start = slice_start.min(len).to_usize();
        let end = (slice_start + slice_len).min(len).to_usize();
        assert_eq!(data, test_data[start..end]);
    }

    /// Test implementation for the test_decode_part test, to be called by both proptest and hardcoded tests
    async fn test_decode_part_async_impl(len: ByteNum, slice_start: ByteNum, slice_len: ByteNum) {
        let test_data = create_test_data(len.to_usize());
        // create a slice encoding the given range
        let (hash, slice) = encode_slice(&test_data, slice_start.0, slice_len.0);
        // SliceIter::print_bao_encoded(len, slice_start..slice_start + slice_len, &slice);

        // create an inner decoder to decode the entire slice
        let mut cursor = futures::io::Cursor::new(&slice);
        let mut validator = SliceValidator::new(&mut cursor, hash, slice_start.0, slice_len.0);
        while let Some(item) = validator.next().await {
            assert!(item.is_ok());
        }
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);

        let mut cursor = futures::io::Cursor::new(&slice);
        let mut reader = SliceReader::new(&mut cursor, hash, slice_start.0, slice_len.0);
        let mut data = vec![];
        reader.read_to_end(&mut data).await.unwrap();
        // check that we have read the entire slice
        assert_eq!(cursor.position(), slice.len() as u64);
        // check that we have read the correct data
        let start = slice_start.min(len).to_usize();
        let end = (slice_start + slice_len).min(len).to_usize();
        assert_eq!(data, test_data[start..end]);
    }

    /// Generate a random size, start and len
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
            // test_decode_all_impl(len);
            test_decode_all_impl(len);
            futures::executor::block_on(test_decode_all_async_impl(len));
        }

        #[test]
        fn test_decode_part((size, start, len) in size_start_len()) {
            // test_decode_part_impl(size, start, len);
            test_decode_part_impl(size, start, len);
            futures::executor::block_on(test_decode_part_async_impl(size, start, len));
        }
    }

    #[test]
    fn test_decode_all_1() {
        let len = ByteNum(12343465);
        // test_decode_all_impl(len);
        test_decode_all_impl(len);
    }

    #[tokio::test]
    async fn test_decode_all_async_1() {
        let len = ByteNum(12343465);
        // test_decode_all_impl(len);
        test_decode_all_async_impl(len).await;
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
    fn test_decode_part_0() {
        test_decode_part_impl(ByteNum(1), ByteNum(0), ByteNum(0));
    }

    #[test]
    fn test_decode_part_1() {
        test_decode_part_impl(ByteNum(2048), ByteNum(0), ByteNum(1024));
        test_decode_part_impl(ByteNum(2048), ByteNum(1024), ByteNum(1024));
        test_decode_part_impl(ByteNum(4096), ByteNum(0), ByteNum(1024));
        test_decode_part_impl(ByteNum(4096), ByteNum(1024), ByteNum(1024));
    }

    #[tokio::test]
    async fn test_decode_part_async_1() {
        test_decode_part_async_impl(ByteNum(2048), ByteNum(0), ByteNum(1024)).await;
        test_decode_part_async_impl(ByteNum(2048), ByteNum(1024), ByteNum(1024)).await;
        test_decode_part_async_impl(ByteNum(4096), ByteNum(0), ByteNum(1024)).await;
        test_decode_part_async_impl(ByteNum(4096), ByteNum(1024), ByteNum(1024)).await;
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
}
