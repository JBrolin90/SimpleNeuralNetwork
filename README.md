# BackPropagation Neural Network (C#)

This project implements a simple neural network with backpropagation in C#. It demonstrates the basics of feedforward and backpropagation learning, using custom activation functions and a modular architecture.

## Features
- Customizable neural network layers and nodes
- Support for different activation functions (SoftPlus, Sigmoid, Unit)
- Manual weight and bias initialization
- Simple test harness for evaluating network performance

## Project Structure
- `Program.cs`: Entry point, demonstrates network creation, prediction, and testing.
- `NNLib/Node.cs`: Defines the `Node` class and activation functions.
- `NNLib/Layer.cs`: Defines the `Layer` class.
- `NNLib/NeuralNetwork.cs`: Implements the neural network logic.
- `NNLib/TestNeuralNetwork.cs`: Provides a simple test harness for the network.

## Example Usage
The main program (`Program.cs`) creates a neural network, predicts outputs for sample inputs, and evaluates the sum of squared residuals (SSR):

```csharp
Console.WriteLine("Hello, BackPropagation learners!");

// Example input and observed output
double[] inputs = [0, 0.5, 1];
double[] observed = [0, 1, 0];

// Network weights, biases, and activation functions
// ... see Program.cs for details ...

NeuralNetwork nn = new(...);
double[] output0 = nn.Predict([0]);
double[] output1 = nn.Predict([0.5]);
double[] output2 = nn.Predict([1.0]);
Console.WriteLine($"Outputs: {output0[0]}, {output1[0]}, {output2[0]}");

// Test the network
TestNeuralNetwork testNN = new(...);
testNN.Test(inputs, observed);
Console.WriteLine($"SSR: {testNN.SSR}");
```

## Note:
While the project demonstrates the structure and core concepts of a neural network with backpropagation, the backpropagation (weight and bias update) logic is not yet fully implemented. The codebase provides a solid modular foundation with dynamic network sizing, customizable activation functions, and a test harness, but the actual learning (gradient descent and parameter updates) is either incomplete or only partially functional.

As such, the network may not learn from data as expected until the backpropagation and update routines are fully developed. This project is ideal for educational exploration and as a starting point for implementing and experimenting with neural network training algorithms.


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
