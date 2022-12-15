use std::ops::Add;

use tch::nn::{Conv2D, conv2d, ConvConfig, Module};
use tch::Tensor;

mod head;
mod basic;
mod bottleneck;

#[derive(Debug)]
pub struct HeadBlock {
    conv: Conv2D,
}

#[derive(Debug)]
pub struct BasicBlock {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
}



#[derive(Debug)]
pub struct Bottleneck {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
    conv4: Conv2D,
}
