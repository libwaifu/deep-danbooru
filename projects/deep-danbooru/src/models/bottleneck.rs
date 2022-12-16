use super::*;

impl Module for Bottleneck {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let ls = xs.apply(&self.conv1)
            .relu()
            .apply(&self.conv2)
            .relu()
            .apply(&self.conv3);
        xs.apply(&self.conv4).add(ls).relu()
    }
}

impl Bottleneck {
    pub fn new(vs: tch::nn::Path, in_channels: i64, out_channels: i64) -> Self {
        let mut config = ConvConfig::default();
        config.padding = 1;
        let conv1 = conv2d(&vs / "conv2d_l1", in_channels, out_channels / 4, 1, Default::default());
        let conv2 = conv2d(&vs / "conv2d_l2", out_channels / 4, out_channels / 4, 3, config);
        let conv3 = conv2d(&vs / "conv2d_l3", out_channels / 4, out_channels, 1, Default::default());
        let conv4 = conv2d(&vs / "conv2d_r1", in_channels, out_channels, 1, Default::default());
        Self { conv1, conv2, conv3, conv4 }
    }
}