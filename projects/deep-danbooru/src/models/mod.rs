use tch::nn::Module;
use tch::Tensor;

mod head;
mod basic;

#[derive(Debug)]
pub struct HeadBlock {

}

#[derive(Debug)]
pub struct BasicBlock {}

#[derive(Debug)]
pub struct Bottleneck {}

impl Module for BasicBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        todo!()
    }
}


impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Tensor {
        todo!()
    }
}