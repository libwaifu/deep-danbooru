use std::{cmp::Ordering, path::Path, sync::Arc};

use image::{imageops::FilterType, DynamicImage, Rgb32FImage};
use itertools::Itertools;
use ndarray::{Array2, ArrayD};
use ort::{tensor::InputTensor, Environment, ExecutionProvider, OrtResult, Session, SessionBuilder};

pub mod tags2019;
pub mod tags2020;
pub mod tags2021;

#[derive(Debug, Clone, Copy)]
pub enum DeepDanbooruPreProgress {
    Resize,
    Crop,
}

pub struct DeepDanbooru {
    session: Session,
    pre_process: DeepDanbooruPreProgress,
    tags: &'static [&'static str],
}

impl DeepDanbooru {
    pub fn new(runtime: &Arc<Environment>, model: &Path) -> OrtResult<Self> {
        let session = SessionBuilder::new(&runtime)?
            .with_execution_providers(&[ExecutionProvider::cuda(), ExecutionProvider::cpu()])?
            .with_model_from_file(model)?;
        Ok(Self { session, pre_process: DeepDanbooruPreProgress::Resize, tags: &[] })
    }
    pub fn set_pre_process(&mut self, pre_process: DeepDanbooruPreProgress) {
        self.pre_process = pre_process;
    }
    pub fn set_tags(&mut self, tags: &'static [&'static str]) {
        self.tags = tags;
    }
    pub fn predict(&self, image: &DynamicImage) -> OrtResult<Vec<(&'static str, f32)>> {
        let post = match self.pre_process {
            DeepDanbooruPreProgress::Resize => image.resize_exact(512, 512, FilterType::CatmullRom),
            DeepDanbooruPreProgress::Crop => image.crop_imm(0, 0, 512, 512),
        };
        let input = one_image_to_tensor(post.to_rgb32f());
        let out = self.session.run(&[input])?;
        let array = out.first().unwrap().try_extract()?;
        let similarity: Array2<f32> = array.view().to_owned().into_dimensionality().unwrap();
        let similarity = similarity.into_raw_vec();
        let out = similarity
            .iter()
            .zip(self.tags.iter())
            .sorted_by(|l, r| l.0.partial_cmp(r.0).unwrap_or(Ordering::Equal))
            .rev()
            .map(|(k, v)| (*v, *k))
            .collect_vec();
        Ok(out)
    }
}

fn one_image_to_tensor(image: Rgb32FImage) -> InputTensor {
    let shape = vec![1, image.width() as usize, image.height() as usize, 3];
    let array = ArrayD::from_shape_vec(shape, image.as_raw().to_vec()).unwrap();
    InputTensor::FloatTensor(array)
}
