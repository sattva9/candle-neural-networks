use candle_core::{DType, Device, Error, Tensor};
use candle_nn::VarMap;
use candle_nn::{linear, ops::softmax, Linear, Module, VarBuilder};

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

fn num_trainable_params(varmap: &VarMap) -> usize {
    let mut total_params = 0;

    for var in varmap.all_vars().iter() {
        let tensor = var.as_tensor();
        total_params += tensor.elem_count();
    }

    total_params
}

fn main() -> Result<(), candle_core::Error> {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = NeuralNetwork::new(50, 3, vb)?;
    println!("{:#?}", model);
    println!(
        "Total number of trainable model parameters: {}",
        num_trainable_params(&varmap)
    );
    println!("{}", model.layer_1.weight());
    println!("{:?}", model.layer_1.weight().shape());

    let x = Tensor::rand(0f32, 1f32, (1, 50), &device)?;
    let output = model.forward(&x)?;
    println!("{}", output);

    let output = model.forward(&x)?;
    let probabilities = softmax(&output, 1)?;
    println!("{}", probabilities);

    Ok(())
}
