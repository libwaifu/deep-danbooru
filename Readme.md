Deep Danbooru for rust
======================

Multi-labels anime image classification without python.

### Usage

```rust, ignore
let runtime = Arc::new(Environment::builder().build()?);
let mut model = DeepDanbooru::new(&runtime, model.as_ref())?;
model.set_tags(TAGS2021);
let image = Reader::open(image.as_ref()).unwrap().decode().unwrap();
let result = model.predict(&image).unwrap();
```

### Develop

- Clone with models

```shell
# https
git clone --recursive https://github.com/libwaifu/deep-danbooru.git
# git
git clone --recursive git@github.com:libwaifu/deep-danbooru.git
```

or download manually: [deep-danbooru/models](https://huggingface.co/oovm/deep-danbooru/tree/main/models)

- Pull latest models

```shell
git pull && git submodule update --remote
```

