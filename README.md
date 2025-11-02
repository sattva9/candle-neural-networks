# Neural Networks with Candle

A hands-on introduction to Neural Networks in Rust using [Candle](https://github.com/huggingface/candle). Covers everything from tensors to building and training your first neural network.

> Based on Appendix A (the PyTorch introduction) from [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch) by [Sebastian Raschka](https://sebastianraschka.com/), with the Python/PyTorch code ported to Rust/Candle.

## Run commands

```bash
cargo run --bin concrete_example         # Computation graphs
cargo run --bin computing_gradients      # Automatic differentiation
cargo run --bin neural_network           # Build a neural network
cargo run --bin data_loaders_and_training # Train and save model
cargo run --bin loading_model            # Load and use saved model
```

See the [blog post](https://pranitha.dev/posts/neural-networks-with-candle/) for detailed explanations.
