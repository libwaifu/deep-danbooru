use super::*;


impl HeadBlock {
    pub fn new(vs: tch::nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let mut config = ConvConfig::default();
        config.padding = 1;
        let conv = conv2d(vs / "conv2d", in_channels, out_channels, 3, config);
        Self { conv }
    }
}

impl Module for HeadBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv).relu().max_pool2d(&[3, 3], &[2, 2], &[1, 1], &[0, 0], false)
    }
}
