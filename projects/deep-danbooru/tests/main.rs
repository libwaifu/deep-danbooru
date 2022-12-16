use std::path::PathBuf;
use image::imageops::FilterType;

use image::io::Reader;
use image::{Rgb32FImage, RgbImage};
use ndarray::ArrayD;
use ort::{Environment, ExecutionProvider, OrtResult, SessionBuilder};
use ort::tensor::InputTensor;

#[test]
fn ready() {
    println!("it works!")
}

#[test]
fn test() -> OrtResult<()> {
    println!("{}", env!("ORT_STRATEGY"));
    println!("{}", env!("ORT_DYLIB_PATH"));
    let projects = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
    println!("{:?}", projects.display());
    let mut env = Environment::default().into_arc();
    let session = SessionBuilder::new(&env)
        .unwrap()
        .with_execution_providers(&[ExecutionProvider::cuda(), ExecutionProvider::cpu()])
        .unwrap()
        .with_model_from_file(
            projects.join("deep-danbooru-models/models/deepdanbooru2020.onnx")
        )
        .unwrap();
    let image = Reader::open(projects.join("deep-danbooru/tests/105715609_p0_master1200.jpg")).unwrap().decode().unwrap();

    let array = make_input_tensor(&image.resize_exact(512, 512, FilterType::CatmullRom).to_rgb32f());


    // let inputs = vec![array];
    let out = session.run(&[array]).unwrap();
    // let index = out;

    println!("{:?}", out);
    Ok(())
}

pub fn make_input_tensor(image: &Rgb32FImage) -> InputTensor {
    let shape = vec![1, image.width() as usize, image.height() as usize, 3];
    let array = ArrayD::from_shape_vec(shape, image.as_raw().to_vec()).unwrap();
    InputTensor::FloatTensor(array)
}