# BackPropagation Neural Network (C#)

This project implements a simple neural network with backpropagation in C#. It demonstrates the basics of feedforward and backpropagation learning, using custom activation functions and a modular architecture.

## Features
- Customizable neural network layers and nodes
- Support for multiple activation functions (SoftPlus, Sigmoid, Unit, Tanh, ReLU)
- Manual weight and bias initialization
- Working gradient descent implementation for network training
- Sum of Squared Residuals (SSR) calculation for error measurement

## Project Structure
- `Program.cs`: Entry point, demonstrates network creation, training, and prediction.
- `NNLib/Node.cs`: Defines the `Node` class with forward and backward propagation logic.
- `NNLib/Layer.cs`: Defines the `Layer` class and layer factory for creating specialized layers.
- `NNLib/InputLayer.cs`: Implementation of the input layer.
- `NNLib/OutputLayer.cs`: Implementation of the output layer.
- `NNLib/NeuralNetwork.cs`: Base neural network implementation with feedforward logic.
- `NNLib/TestNeuralNetwork.cs`: Extends the neural network with training capabilities.
- `NNLib/ActivationFunction.cs`: Collection of activation functions and their derivatives.

## Example Usage
The main program (`Program.cs`) creates a neural network, trains it on sample inputs, and shows the resulting error:

```csharp
Console.WriteLine("Hello, BackPropagation learners!");

// Example input and observed output
double[] inputs = [0, 0.5, 1];
double[] observed = [0, 1, 0];

// Network weights, biases, and activation functions setup
double[][][] weights = [
    [],
    [[2.74], [-1.13]],
    [[0.36, 0.63]],
    []
];
double[][][] biases = [[], [[0], [0]], [[0]], []];
double[][] ys = [[0,0],[0,0],[0, 0], [0, 0]];

Func<double, double>[] activationFunctions = [
    ActivationFunctions.Unit,
    ActivationFunctions.SoftPlus,
    ActivationFunctions.Unit,
    ActivationFunctions.Unit
];

// Train the network
TestNeuralNetwork testNN = new(new LayerFactory(), new NodeFactory(),
    weights, biases, ys, 0.01, activationFunctions);
for (int i = 0; i < 20000; i++)
{
    testNN.Train(inputs, observed);
}
Console.WriteLine($"SSR: {testNN.SSR}");
``` ## Implementation Details

The neural network implementation follows these key principles:

1. **Modular Architecture**:
   - Neural networks are composed of layers
   - Layers contain nodes
   - Factory pattern used for creating specialized layers and nodes

2. **Forward Propagation**:
   - Input is fed through the network layer by layer
   - Each node applies weights, adds bias, and applies an activation function
   - Output is collected from the final layer

3. **Backpropagation**:
   - Error is calculated as the difference between predicted and expected output
   - Error is propagated backward through the network using the chain rule
   - Weight and bias gradients are calculated for each node
   - Gradient descent is used to update weights and biases

4. **Chain Rule Implementation**:
   - Layer and node implementations handle gradient calculations
   - `GetWeightChainFactor` and `GetBiasChainFactor` methods implement chain rule logic
   - NodeSteps stores weight and bias updates during backpropagation

## How to Run
1. **Build the project:**
   ```bash
   dotnet build
   ```
2. **Run the program:**
   ```bash
   dotnet run
   ```

## Requirements
- .NET 9.0 SDK or later
- Linux, Windows, or macOS

## License
This project is for educational purposes.
