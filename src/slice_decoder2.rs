use crate::tree::*;
use futures::AsyncRead;
use std::io::{self, Read};

pub struct DecoderInner<R> {
    inner: R,
    // start block (inclusive)
    start_block: BlockNum,
    // end block (exclusive)
    end_block: BlockNum,
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
    pub fn new(
        inner: R,
        hash: blake3::Hash,
        start_block: BlockNum,
        end_block: BlockNum,
        len: ByteNum,
    ) -> Self {
        Self {
            inner,
            start_block,
            end_block,
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
                if self.end_block <= position {
                    // we only need to go to the left
                    self.stack.push(lh);
                    self.offset = left_child(self.offset).unwrap();
                } else if self.start_block >= position {
                    // we only need to go to the right
                    self.stack.push(rh);
                    self.offset = right_child(self.offset).unwrap();
                } else {
                    // we need to visit both, but first left
                    self.stack.push(rh);
                    self.stack.push(lh);
                    self.offset = left_child(self.offset).unwrap();
                }
                break Ok(64);
            } else {
                let ls = self.read_and_validate_leaf()?;
                self.offset = self.offset + 2;
                let rs = self.read_and_validate_leaf()?;
                let offset0 = self.offset;
                while parent(self.offset) < self.offset {
                    self.offset = parent(self.offset);
                    println!("{:?}", self.offset);
                }
                self.offset = parent(self.offset);
                println!("{:?}", self.offset);
                match right_descendant(self.offset, self.hashes()) {
                    Some(offset) => self.offset = offset,
                    None => break Ok((ls + rs) as u64),
                }
                println!("{:?} => {:?}", offset0, self.offset);
                break Ok((ls + rs) as u64);
            }
        }
    }

    fn read_and_validate_leaf(&mut self) -> io::Result<usize> {
        let block = index(self.offset);
        if block < self.start_block || block >= self.end_block {
            return Ok(0);
        }
        let size = leaf_size(block, self.block_level(), self.len).to_usize();
        self.inner.read_exact(&mut self.buf[..size])?;
        let hash = hash_block(block, &self.buf[..size], self.block_level(), self.is_root());
        let expected = self.stack.pop().unwrap();
        if hash != expected {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "hash mismatch"));
        } else {
            println!("leaf {:?} validated", block);
        }
        Ok(size)
    }
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
    use bao::encode::SliceExtractor;
    use std::io::{Cursor, Read};

    fn mk_test_data(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i / 1024) as u8).collect()
    }

    fn mk_slice(data: &[u8]) -> (blake3::Hash, Vec<u8>) {
        let (encoded, hash) = bao::encode::encode(data);
        let mut extractor = SliceExtractor::new(Cursor::new(&encoded), 0, data.len() as u64);
        let mut slice = vec![];
        extractor.read_to_end(&mut slice).unwrap();
        (hash, slice)
    }

    #[test]
    fn test_decode() {
        let len = 8190;
        let (hash, slice) = mk_slice(&mk_test_data(len));
        println!("{}", hex::encode(&slice));
        println!("{}", slice.len() - 8 - 64);
        let mut decoder = DecoderInner::new(
            Cursor::new(&slice[8..]),
            hash,
            BlockNum(0),
            BlockNum(8),
            ByteNum(len as u64),
        );
        while let Ok(n) = decoder.advance() {
            println!("read {} bytes", n);
            if n == 0 {
                break;
            }
        }
    }
}
