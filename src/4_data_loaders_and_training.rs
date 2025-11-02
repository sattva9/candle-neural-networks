use candle_core::{DType, Device, Error, Tensor};
use candle_nn::{
    Linear, Module, Optimizer, SGD, VarBuilder, VarMap, linear, loss::cross_entropy, ops::softmax,
};
use rand::seq::SliceRandom;

#[derive(Debug)]
struct NeuralNetwork {
    layer_1: Linear,
    layer_2: Linear,
    output_layer: Linear,
}

impl NeuralNetwork {
    fn new(num_inputs: usize, num_outputs: usize, vb: VarBuilder) -> Result<Self, Error> {
        Ok(Self {
            // 1st hidden layer
            layer_1: linear(num_inputs, 30, vb.pp("l1"))?,
            // 2nd hidden layer
            layer_2: linear(30, 20, vb.pp("l2"))?,
            // output layer
            output_layer: linear(20, num_outputs, vb.pp("out"))?,
        })
    }
}

impl Module for NeuralNetwork {
    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // Pass through 1st hidden layer with ReLU activation
        let x = self.layer_1.forward(x)?;
        let x = x.relu()?;

        // Pass through 2nd hidden layer with ReLU activation
        let x = self.layer_2.forward(&x)?;
        let x = x.relu()?;

        // Output layer (logits)
        self.output_layer.forward(&x)
    }
}

struct Dataset {
    features: Tensor,
    labels: Tensor,
}

impl Dataset {
    fn new(x: Tensor, y: Tensor) -> Self {
        Self {
            features: x,
            labels: y,
        }
    }

    fn get_item(&self, index: usize) -> Result<(Tensor, Tensor), Error> {
        let one_x = self.features.get(index)?;
        let one_y = self.labels.get(index)?;
        Ok((one_x, one_y))
    }

    fn len(&self) -> Result<usize, Error> {
        self.labels.dims1()
    }
}

struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    indices: Vec<usize>,
    drop_last: bool,
}

impl DataLoader {
    fn new(
        dataset: Dataset,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
    ) -> Result<Self, Error> {
        let len = dataset.len()?;
        let mut indices: Vec<usize> = (0..len).collect();

        if shuffle {
            indices.shuffle(&mut rand::rng());
        }

        Ok(Self {
            dataset,
            batch_size,
            indices,
            drop_last,
        })
    }

    fn iter(&self) -> DataLoaderIter {
        DataLoaderIter {
            dataset: &self.dataset,
            indices: &self.indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            current: 0,
        }
    }

    fn total_batches(&self) -> usize {
        if self.drop_last {
            self.indices.len() / self.batch_size
        } else {
            (self.indices.len() + self.batch_size - 1) / self.batch_size
        }
    }
}

struct DataLoaderIter<'a> {
    dataset: &'a Dataset,
    indices: &'a [usize],
    batch_size: usize,
    drop_last: bool,
    current: usize,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor), Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current..end];

        // Drop last incomplete batch if specified
        if self.drop_last && batch_indices.len() < self.batch_size {
            return None;
        }

        self.current = end;

        let mut batch_features = Vec::new();
        let mut batch_labels = Vec::new();

        for &idx in batch_indices {
            match self.dataset.get_item(idx) {
                Ok((features, labels)) => {
                    batch_features.push(features);
                    batch_labels.push(labels);
                }
                Err(e) => return Some(Err(e)),
            }
        }

        match (
            Tensor::stack(&batch_features, 0),
            Tensor::stack(&batch_labels, 0),
        ) {
            (Ok(batch_x), Ok(batch_y)) => Some(Ok((batch_x, batch_y))),
            (Err(e), _) | (_, Err(e)) => Some(Err(e)),
        }
    }
}

fn compute_accuracy(model: &NeuralNetwork, dataloader: &DataLoader) -> Result<f32, Error> {
    let mut correct = 0;
    let mut total_examples = 0;

    for batch_result in dataloader.iter() {
        let (features, labels) = batch_result?;

        let logits = model.forward(&features)?;
        let predictions = logits.argmax(1)?;

        // Returns a tensor of True/False values
        let compare = predictions.eq(&labels)?;

        // Count number of correct predictions
        let correct_batch = compare.sum_all()?;
        // Convert Tensor to u8 value
        correct += correct_batch.to_vec0::<u8>()?;

        total_examples += compare.elem_count();
    }

    Ok(correct as f32 / total_examples as f32)
}

fn main() -> Result<(), Error> {
    let device = Device::Cpu;

    let x_train = Tensor::new(
        &[
            [-1.2, 3.1],
            [-0.9, 2.9],
            [-0.5, 2.6],
            [2.3, -1.1],
            [2.7, -1.5],
        ],
        &device,
    )?
    .to_dtype(DType::F32)?;
    let y_train = Tensor::new(&[0., 0., 0., 1., 1.], &device)?.to_dtype(DType::U32)?;
    let x_test = Tensor::new(&[[-0.8, 2.8], [2.6, -1.6]], &device)?.to_dtype(DType::F32)?;
    let y_test = Tensor::new(&[0., 1.], &device)?.to_dtype(DType::U32)?;

    // We clone the tensors because Dataset::new takes ownership.
    // This keeps the originals available if needed later.
    // For large datasets, consider using references to avoid cloning overhead.
    let train_ds = Dataset::new(x_train.clone(), y_train.clone());
    let test_ds = Dataset::new(x_test.clone(), y_test.clone());
    println!("{}", train_ds.len()?);

    let train_loader = DataLoader::new(train_ds, 2, true, true)?;
    let test_loader = DataLoader::new(test_ds, 2, false, true)?;

    for (idx, batch_result) in train_loader.iter().enumerate() {
        let (x, y) = batch_result?;
        println!("Batch {}: {}  {}", idx + 1, x, y);
    }

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // The dataset has two features and two classes
    let model = NeuralNetwork::new(2, 2, vb)?;

    // Create optimizer with learning rate 0.5
    let mut optimizer = SGD::new(varmap.all_vars(), 0.5)?;

    let num_epochs = 3;
    for epoch in 0..num_epochs {
        for (batch_idx, batch_result) in train_loader.iter().enumerate() {
            let (features, labels) = batch_result?;

            // Forward pass
            let logits = model.forward(&features)?;
            let loss = cross_entropy(&logits, &labels)?;

            // Backward pass and parameter update
            optimizer.backward_step(&loss)?;

            // Logging
            let loss_val = loss.to_vec0::<f32>()?;
            println!(
                "Epoch: {:03}/{:03} | Batch {:03}/{:03} | Train Loss: {:.2}",
                epoch + 1,
                num_epochs,
                batch_idx + 1,
                train_loader.total_batches(),
                loss_val
            );
        }
    }

    let outputs = model.forward(&x_train)?;
    println!("{}", outputs);

    let probas = softmax(&outputs, 1)?;
    println!("{}", probas);

    let predictions = probas.argmax(1)?;
    println!("predictions: {}", predictions);

    let predictions = outputs.argmax(1)?;
    println!("predictions: {}", predictions);

    println!("{}", predictions.eq(&y_train)?.sum_all()?);

    let accuracy = compute_accuracy(&model, &train_loader)?;
    println!("{}", accuracy);

    let accuracy = compute_accuracy(&model, &test_loader)?;
    println!("{}", accuracy);

    // Save model
    varmap.save("model.safetensors")?;

    Ok(())
}
