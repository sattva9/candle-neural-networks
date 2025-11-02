use candle_core::{Device, Tensor, Var};
use candle_nn::loss::binary_cross_entropy_with_logit;

fn main() -> Result<(), candle_core::Error> {
    let device = Device::Cpu;

    let y = Tensor::new(&[1.], &device)?;
    let x1 = Tensor::new(&[1.1], &device)?;
    let w1 = Var::new(&[2.2], &device)?;
    let b = Var::new(&[0.], &device)?;

    let z = (&x1 * w1.as_tensor() + b.as_tensor())?;

    let loss = binary_cross_entropy_with_logit(&z, &y.flatten_all()?)?;

    // Backward pass - compute gradients
    let grads = loss.backward()?;

    // Get gradients for our parameters
    let grad_l_w1 = grads.get(&w1);
    let grad_l_b = grads.get(&b);
    println!("{}", grad_l_w1.unwrap());
    println!("{}", grad_l_b.unwrap());

    Ok(())
}
