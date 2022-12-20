## Deep Danbooru in rust

Multi-labels anime image classification without python.

### Fast usage

```rust, ignore
# let model_path = projects.join("deep-danbooru-models/models/deepdanbooru-2021.onnx");
# let image_path = projects.join("deep-danbooru-models/tests/pixel-105715609.jpg");
predict_by_danbooru2021(&model_path, &image_path)
```

### Full usage

```rust, ignore
let runtime = Arc::new(Environment::builder().build()?);
let mut model = DeepDanbooru::new(&runtime, model.as_ref())?;
model.set_tags(TAGS2021);
let image = Reader::open(image.as_ref()).unwrap().decode().unwrap();
let result = model.predict(&image).unwrap();
```