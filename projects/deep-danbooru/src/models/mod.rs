use std::ops::Add;

use tch::nn::{Conv2D, conv2d, ConvConfig, Module, VarStore};
use tch::Tensor;

mod head;
mod basic;
mod bottleneck;

#[derive(Debug)]
pub struct DeepDanbooruModel {
    head: HeadBlock,
    bottleneck1: Bottleneck,
    basic1: [BasicBlock; 2],
    bottleneck2: Bottleneck,
    basic2: [BasicBlock; 7],
    bottleneck3: Bottleneck,
    basic3: [Bottleneck; 19],
    bottleneck4: Bottleneck,
    basic4: [Bottleneck; 19],
    bottleneck5: Bottleneck,
    basic5: [BasicBlock; 2],
    bottleneck6: Bottleneck,
    basic6: [BasicBlock; 2],
    tail: TailBlock,
}


#[derive(Debug)]
pub struct HeadBlock {
    conv: Conv2D,
}

pub struct TailBlock {
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


impl Module for DeepDanbooruModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut xs = xs.apply(&self.head);
        xs = xs.apply(&self.bottleneck1);
        for basic in self.basic1.iter() {
            xs = xs.apply(basic);
        }
        xs = xs.apply(&self.bottleneck2);
        for basic in self.basic2.iter() {
            xs = xs.apply(basic);
        }
        xs = xs.apply(&self.bottleneck3);
        for basic in self.basic3.iter() {
            xs = xs.apply(basic);
        }
        xs = xs.apply(&self.bottleneck4);
        for basic in self.basic4.iter() {
            xs = xs.apply(basic);
        }
        xs = xs.apply(&self.bottleneck5);
        for basic in self.basic5.iter() {
            xs = xs.apply(basic);
        }
        xs = xs.apply(&self.bottleneck6);
        for basic in self.basic6.iter() {
            xs = xs.apply(basic);
        }
        xs.apply(&self.tail)
    }
}

impl DeepDanbooruModel {
    pub fn new(vs: &VarStore) -> Self {
        let vs = &vs.root();
        let head = HeadBlock::new(vs / "head", 3, 64);
        let bottleneck1 = Bottleneck::new(vs / "bottleneck_1", 64, 256);
        for i in 1..=2 {
            let path = vs / format!("basic_1_{}", i);
            let basic = BasicBlock::new(&path, 256, 256);
        }
        todo!()
    }
}