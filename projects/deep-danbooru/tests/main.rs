use std::{path::PathBuf, sync::Arc};

use image::io::Reader;
use ort::{Environment, OrtResult};

use deep_danbooru::{DeepDanbooru, Tags2Rust};

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

#[test]
fn write_rust_tags() {
    let text = include_str!("../../deep-danbooru-models/models/deepdanbooru-2021.tags");
    let mut tags = Tags2Rust::new(2021);
    tags.parse(text);
    println!("{:?}", tags.tags.len());
}
