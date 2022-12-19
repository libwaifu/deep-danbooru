use std::{
    fmt::{Debug, Display, Formatter, Write},
    path::{Path, PathBuf},
};

#[derive(Debug)]
pub struct Tags2Rust {
    pub year: u32,
    pub tags: Vec<String>,
}

impl Tags2Rust {
    pub fn new(year: u32) -> Self {
        Self { year, tags: vec![] }
    }

    pub fn parse(&mut self, text: &str) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match line {
                "0_0" => self.add("0_0"),
                _ => self.parse_line(line),
            }
        }
    }
    pub fn add(&mut self, tag: &str) {
        self.tags.push(tag.to_string());
    }
    pub fn parse_line(&mut self, tag: &str) {
        let mut normed = String::new();
        for char in tag.chars() {
            match char {
                '_' => normed.push(' '),
                _ => normed.push(char),
            }
        }
        self.tags.push(normed);
    }
    pub fn file_path(&self, path: &Path) -> PathBuf {
        assert!(path.is_dir(), "TODO");
        path.join(format!("tags{}.rs", self.year))
    }
}

impl Display for Tags2Rust {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "pub const TAGS{}: &'static [&'static str; {}] = ", self.year, self.tags.len());
    }
}
