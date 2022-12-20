use std::{path::Path, sync::Arc};

use image::io::Reader;
use ort::{Environment, OrtResult};

use crate::{DeepDanbooru, TAGS2021};

/// One-time prediction of tags in pictures, suitable for command line usage.
pub fn predict_by_danbooru2021<M, I>(model: M, image: I) -> OrtResult<Vec<(&'static str, f32)>>
where
    M: AsRef<Path>,
    I: AsRef<Path>,
{
    let runtime = Arc::new(Environment::builder().build()?);
    let mut model = DeepDanbooru::new(&runtime, model.as_ref())?;
    model.set_tags(TAGS2021);
    let image = Reader::open(image.as_ref()).unwrap().decode().unwrap();
    let result = model.predict(&image).unwrap();
    Ok(result)
}
