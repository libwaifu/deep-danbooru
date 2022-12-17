use std::{path::Path, sync::Arc};

use image::{imageops::FilterType, DynamicImage, Rgb32FImage};
use ndarray::ArrayD;
use ort::{tensor::InputTensor, Environment, ExecutionProvider, OrtResult, Session, SessionBuilder};

mod tags2019;
mod tags2020;
mod tags2021;
mod tags2rust;

#[derive(Debug, Clone, Copy)]
pub enum DeepDanbooruPreProgress {
    Resize,
    Crop,
}

pub struct DeepDanbooru {
    session: Session,
    pre_process: DeepDanbooruPreProgress,
}

impl DeepDanbooru {
    pub fn new(runtime: &Arc<Environment>, model: &Path) -> OrtResult<Self> {
        let session = SessionBuilder::new(&runtime)?
            .with_execution_providers(&[ExecutionProvider::cuda(), ExecutionProvider::cpu()])?
            .with_model_from_file(model)?;
        Ok(Self { session, pre_process: DeepDanbooruPreProgress::Resize })
    }
    pub fn set_pre_process(&mut self, pre_process: DeepDanbooruPreProgress) {
        self.pre_process = pre_process;
    }
    pub fn predict(&self, image: &DynamicImage) -> OrtResult<Vec<f32>> {
        let post = match self.pre_process {
            DeepDanbooruPreProgress::Resize => image.resize_exact(512, 512, FilterType::CatmullRom),
            DeepDanbooruPreProgress::Crop => image.crop_imm(0, 0, 512, 512),
        };
        let input = make_input_tensor(&post.to_rgb32f());
        let out = self.session.run(&[input])?;
        println!("{:?}", out);
        todo!()
    }
}

fn make_input_tensor(image: &Rgb32FImage) -> InputTensor {
    let shape = vec![1, image.width() as usize, image.height() as usize, 3];
    let array = ArrayD::from_shape_vec(shape, image.as_raw().to_vec()).unwrap();
    InputTensor::FloatTensor(array)
}
