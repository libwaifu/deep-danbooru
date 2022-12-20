#![doc = include_str!("../Readme.md")]

pub use image::DynamicImage;

pub use crate::{
    models::{tags2019::TAGS2019, tags2021::TAGS2021, DeepDanbooru},
    tags2rust::Tags2Rust,
    utils::predict_by_danbooru2021,
};

mod models;
mod tags2rust;
mod utils;
