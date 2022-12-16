use super::*;

impl Module for BasicBlock {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            .apply(&self.conv3)
            .relu()
            .add(xs)
            .relu()
    }
}

impl BasicBlock {
    pub fn new(vs: &tch::nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let mut config = ConvConfig::default();
        config.padding = 1;
        let conv1 = conv2d(vs / "conv2d_1", in_channels, out_channels, 1, Default::default());
        let conv2 = conv2d(vs / "conv2d_2", out_channels, out_channels, 3, config);
        let conv3 = conv2d(vs / "conv2d_3", out_channels, out_channels, 1, Default::default());
        Self { conv1, conv2, conv3 }
    }
}