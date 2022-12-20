use itertools::Itertools;
use std::{fs::read_to_string, path::PathBuf};

use ort::OrtResult;

use deep_danbooru::{predict_by_danbooru2021, Tags2Rust};

#[test]
fn ready() {
    println!("it works!")
}

#[test]
fn test_1gril() -> OrtResult<()> {
    let projects = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap();
    let model_path = projects.join("deep-danbooru-models/models/deepdanbooru-2021.onnx");
    let image_path = projects.join("deep-danbooru-models/tests/pixel-105715609.jpg");
    let tags = predict_by_danbooru2021(model_path, image_path)?;
    println!("{:#?}", tags.iter().take(10).collect_vec());
    Ok(())
}

#[test]
fn write_rust_tags() -> std::io::Result<()> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models").canonicalize()?;
    let danbooru = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../deep-danbooru-models/models").canonicalize()?;
    // 2019
    let mut tags = Tags2Rust::new(2019);
    tags.parse(&read_to_string(danbooru.join("deepdanbooru-2019.tags"))?);
    tags.write_file(&here)?;
    // 2021
    let mut tags = Tags2Rust::new(2021);
    tags.parse(&read_to_string(danbooru.join("deepdanbooru-2021.tags"))?);
    tags.write_file(&here)?;
    Ok(())
}
