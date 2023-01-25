//! Define a number of newtypes and operations on these newtypes
//!
//! Most operations are concerned with node indexes in an in order traversal of a binary tree.
use std::ops::{Add, Div, Mul, Range, Sub};

/// A newtype for a thing that can be conveniently be use as an index
///
/// The intention is not to make the newtype completely foolproof, but to make it
/// convenient to use while still providing some safety by making conversions explicit.
macro_rules! index_newtype {
    (
        $(#[$outer:meta])*
        pub struct $name:ident(pub $wrapped:ty);
    ) => {
        $(#[$outer])*
        #[repr(transparent)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub $wrapped);


        impl Mul<$wrapped> for $name {
            type Output = $name;

            fn mul(self, rhs: $wrapped) -> Self::Output {
                $name(self.0 * rhs)
            }
        }

        impl Div<$wrapped> for $name {
            type Output = $name;

            fn div(self, rhs: $wrapped) -> Self::Output {
                $name(self.0 / rhs)
            }
        }

        impl Sub<$wrapped> for $name {
            type Output = $name;

            fn sub(self, rhs: $wrapped) -> Self::Output {
                $name(self.0 - rhs)
            }
        }

        impl Sub<$name> for $name {
            type Output = $name;

            fn sub(self, rhs: $name) -> Self::Output {
                $name(self.0 - rhs.0)
            }
        }

        impl Add<$wrapped> for $name {
            type Output = $name;

            fn add(self, rhs: $wrapped) -> Self::Output {
                $name(self.0 + rhs)
            }
        }

        impl Add<$name> for $name {
            type Output = $name;

            fn add(self, rhs: $name) -> Self::Output {
                $name(self.0 + rhs.0)
            }
        }

        impl PartialEq<$wrapped> for $name {
            fn eq(&self, other: &$wrapped) -> bool {
                self.0 == *other
            }
        }

        impl PartialEq<$name> for $wrapped {
            fn eq(&self, other: &$name) -> bool {
                *self == other.0
            }
        }

        impl PartialOrd<$wrapped> for $name {
            fn partial_cmp(&self, other: &$wrapped) -> Option<std::cmp::Ordering> {
                self.0.partial_cmp(other)
            }
        }

        impl $name {

            /// Convert to usize or panic if it doesn't fit.
            pub fn to_usize(self) -> usize {
                usize::try_from(self.0).expect("usize overflow")
            }
        }
    }
}

pub(crate) const BLAKE3_CHUNK_SIZE: u64 = 1024;

index_newtype! {
    /// a number of leaf blocks with its own hash
    pub struct BlockNum(pub u64);
}

/// A tree level. 0 is for leaves, 1 is for the first level of branches, etc.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TreeLevel(pub u32);

pub type NodeNum = u64;

/// Root offset given a number of leaves.
pub(crate) fn root(leafs: BlockNum) -> NodeNum {
    root0(leafs.0)
}

fn root0(leafs: u64) -> u64 {
    leafs.next_power_of_two() - 1
}

/// Level for an offset. 0 is for leaves, 1 is for the first level of branches, etc.
pub(crate) fn level(offset: NodeNum) -> u32 {
    level0(offset)
}

fn level0(offset: u64) -> u32 {
    (!offset).trailing_zeros()
}

pub(crate) fn blocks(len: u64) -> BlockNum {
    BlockNum(blocks0(len))
}

fn blocks0(len: u64) -> u64 {
    let block_size = 1024;
    len / block_size + if len % block_size == 0 { 0 } else { 1 }
}

pub(crate) fn num_hashes(blocks: BlockNum) -> NodeNum {
    num_hashes0(blocks.0)
}

fn num_hashes0(blocks: u64) -> u64 {
    if blocks > 0 {
        blocks * 2 - 1
    } else {
        1
    }
}

fn span0(offset: u64) -> u64 {
    1 << (!offset).trailing_zeros()
}

pub fn left_child(offset: NodeNum) -> Option<NodeNum> {
    let span = span0(offset);
    if span == 1 {
        None
    } else {
        Some(offset - span / 2)
    }
}

fn right_child(offset: NodeNum) -> Option<NodeNum> {
    let span = span0(offset);
    if span == 1 {
        None
    } else {
        Some(offset + span / 2)
    }
}

/// Get a valid right descendant for an offset
pub(crate) fn right_descendant(offset: NodeNum, len: NodeNum) -> Option<NodeNum> {
    let mut offset = right_child(offset)?;
    while offset >= len {
        offset = left_child(offset)?;
    }
    Some(offset)
}

/// Get the chunk index for an offset
pub(crate) fn index(offset: NodeNum) -> BlockNum {
    BlockNum(offset / 2)
}

pub(crate) fn range(offset: NodeNum) -> Range<NodeNum> {
    let r = range0(offset);
    r.start..r.end
}

fn range0(offset: u64) -> Range<u64> {
    let span = span0(offset);
    offset + 1 - span..offset + span
}

#[cfg(test)]
mod tests {
    use std::cmp::{max, min};

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn test_right_descendant() {
        for i in 1..11 {
            println!(
                "valid_right_child({}, 9), {:?}",
                i,
                right_descendant(i, 9)
            );
        }
    }

    #[test]
    fn test_span() {
        for i in 0..10 {
            println!("assert_eq!(span({}), {})", i, span0(i))
        }
    }

    #[test]
    fn test_level() {
        for i in 0..10 {
            println!("assert_eq!(level({}), {})", i, level0(i))
        }
        assert_eq!(level0(0), 0);
        assert_eq!(level0(1), 1);
        assert_eq!(level0(2), 0);
        assert_eq!(level0(3), 2);
    }

    #[test]
    fn test_range() {
        for i in 0..8 {
            println!("{} {:?}", i, range0(i));
        }
    }

    #[test]
    fn test_root() {
        assert_eq!(root0(0), 0);
        assert_eq!(root0(1), 0);
        assert_eq!(root0(2), 1);
        assert_eq!(root0(3), 3);
        assert_eq!(root0(4), 3);
        assert_eq!(root0(5), 7);
        assert_eq!(root0(6), 7);
        assert_eq!(root0(7), 7);
        assert_eq!(root0(8), 7);
        assert_eq!(root0(9), 15);
        assert_eq!(root0(10), 15);
        assert_eq!(root0(11), 15);
        assert_eq!(root0(12), 15);
        assert_eq!(root0(13), 15);
        assert_eq!(root0(14), 15);
        assert_eq!(root0(15), 15);
        assert_eq!(root0(16), 15);
        assert_eq!(root0(17), 31);
        assert_eq!(root0(18), 31);
        assert_eq!(root0(19), 31);
        assert_eq!(root0(20), 31);
        assert_eq!(root0(21), 31);
        assert_eq!(root0(22), 31);
        assert_eq!(root0(23), 31);
        assert_eq!(root0(24), 31);
        assert_eq!(root0(25), 31);
        assert_eq!(root0(26), 31);
        assert_eq!(root0(27), 31);
        assert_eq!(root0(28), 31);
        assert_eq!(root0(29), 31);
        assert_eq!(root0(30), 31);
        assert_eq!(root0(31), 31);
        for i in 1..32 {
            println!("assert_eq!(root0({}),{});", i, root0(i))
        }
    }
}
