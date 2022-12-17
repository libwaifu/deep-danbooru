use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use image::{imageops::FilterType, io::Reader, DynamicImage, Rgb32FImage};
use ndarray::ArrayD;
use ort::{tensor::InputTensor, Environment, ExecutionProvider, OrtResult, Session, SessionBuilder};

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
    let runtime = Arc::new(Environment::builder().build()?);
    let model = DeepDanbooru::new(&runtime, projects.join("deep-danbooru-models/models/deepdanbooru-2021.onnx").as_path())?;
    let image = Reader::open(projects.join("deep-danbooru-models/tests/pixel-105715609.jpg")).unwrap().decode().unwrap();
    let result = model.predict(&image).unwrap();
    println!("{:?}", result);
    Ok(())
}
