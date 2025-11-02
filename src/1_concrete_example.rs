use candle_core::{Device, Tensor};
use candle_nn::loss::binary_cross_entropy_with_logit;
use candle_nn::ops::sigmoid;

fn main() -> Result<(), candle_core::Error> {
    let device = Device::Cpu;

    // True label
    let y = Tensor::new(&[1.], &device)?;

    // Input feature
    let x1 = Tensor::new(&[1.1], &device)?;

    // Weight parameter
    let w1 = Tensor::new(&[2.2], &device)?;

    // Bias unit
    let b = Tensor::new(&[0.], &device)?;

    // Net input
    let z = (x1 * w1 + b)?;

    // Activation and output (computed for illustration, matching the diagram)
    let a = sigmoid(&z)?;

    // Loss: uses z (logits) directly, as binary_cross_entropy_with_logit
    // applies sigmoid internally for numerical efficiency
    let loss = binary_cross_entropy_with_logit(&z, &y)?;
    println!("{}", loss);

    Ok(())
}
