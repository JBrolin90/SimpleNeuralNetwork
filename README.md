# Simple Neural Network with Backpropagation (C#)

This project implements a comprehensive neural network library with backpropagation in C#. It demonstrates advanced feedforward and backpropagation learning algorithms using custom activation functions, modular architecture, and extensive testing frameworks.

## Solution Structure

The solution (`BackPropagation.sln`) contains multiple projects:

- **ConsoleApp** (`ConsoleApp/`): Main application with example implementations
- **SimpleNeuralNetwork.Tests**: Comprehensive xUnit test suite
- **BackPropagation.Tests**: Additional specialized test implementations

## Features
- Modular neural network architecture with factory patterns
- Support for multiple activation functions (SoftPlus, Sigmoid, Unit, Tanh, ReLU)
- Flexible layer and neuron implementations
- Advanced training algorithms with backpropagation
- Multiple loss function implementations
- Comprehensive input processing utilities
- Gradient descent optimization
- Extensive test coverage with xUnit framework

## Project Structure

### Main Application (`ConsoleApp/`)
- `Program.cs`: Entry point with various test implementations
- `AdderTest.cs`, `AlgebraTest.cs`, `MinimalTest.cs`: Example implementations and tests
- `OneNeuronTest.cs`, `Linear2LayersTest.cs`, `TwoLayerAdder.cs`: Neural network test scenarios
- `Multiplier.cs`: Current active implementation
- `StatQuestPart2.cs`, `StatQuestPart2Random.cs`: Statistical learning examples

### Neural Network Library (`ConsoleApp/NNLib/`)
- `NeuralNetwork.cs`: Core neural network implementation with interfaces
- `NeuralNetworkTrainer.cs`: Training algorithms and backpropagation logic
- `Layer.cs`: Layer implementation and factory patterns
- `Neuron.cs`: Individual neuron implementation with forward/backward propagation
- `ActivationFunction.cs`: Collection of activation functions and their derivatives
- `InputProcessor.cs`: Input data processing utilities
- `LossFunctions.cs`: Error calculation and loss function implementations
- `NetworkCreator.cs`: Factory for creating neural network architectures
- `Sample.cs`: Data sample structures for training
- `TestNeuralNetwork.cs`: Extended neural network with testing capabilities

### Test Suites
- `SimpleNeuralNetwork.Tests/`: Comprehensive xUnit test suite covering all components
- `BackPropagation.Tests/`: Additional test implementations

## Example Usage

The project includes multiple example implementations demonstrating different neural network scenarios:

### Current Active Example (Multiplier)
```csharp
// Run the multiplier neural network example
new Multiplier().DoIt();
```

### Available Examples
- **Multiplier**: Neural network for multiplication operations
- **AdderTest**: Addition neural network implementation  
- **TwoLayerAdder**: Two-layer architecture for addition
- **Linear2LayersTest**: Linear regression with two layers
- **OneNeuronTest**: Single neuron learning example
- **AlgebraTest**: General algebraic operations
- **MinimalTest**: Simplified neural network demonstration

### Basic Neural Network Setup
```csharp
// Example network configuration
var neuralNetwork = new NeuralNetwork(
    layerFactory, 
    neuronFactory, 
    inputProcessorFactory,
    weights, 
    biases, 
    ys, 
    activationFunctions
);

// Training with NeuralNetworkTrainer
var trainer = new NeuralNetworkTrainer(neuralNetwork, learningRate);
trainer.Train(trainingData, epochs);
``` ## Implementation Details

The neural network implementation follows these key principles:

1. **Modular Architecture**:
   - Neural networks composed of layers using factory patterns
   - Layers contain neurons with flexible implementations
   - Separation of concerns between network structure and training logic
   - Interface-based design for extensibility

2. **Forward Propagation**:
   - Input processing through the InputProcessor
   - Layer-by-layer computation with configurable activation functions
   - Efficient matrix operations for weight calculations
   - Output collection and formatting

3. **Backpropagation**:
   - Error calculation using various loss functions
   - Gradient computation through the chain rule
   - Weight and bias updates via gradient descent
   - Support for different optimization algorithms

4. **Training Infrastructure**:
   - NeuralNetworkTrainer handles the training process
   - Sample-based training data management
   - Configurable learning rates and training parameters
   - Comprehensive error tracking and convergence monitoring

## How to Run

### Building and Running
1. **Build the entire solution:**
   ```bash
   dotnet build BackPropagation.sln
   ```

2. **Run the console application:**
   ```bash
   cd ConsoleApp
   dotnet run
   ```

3. **Run a specific example:**
   ```bash
   # Edit Program.cs to uncomment the desired example
   # Current active: new Multiplier().DoIt();
   dotnet run
   ```

### Available Examples
To run different examples, edit `ConsoleApp/Program.cs` and uncomment the desired line:
- `new StatQuestPart2()` - Statistical learning demonstration
- `new AlgebraTest().DoIt()` - Algebraic operations
- `new MinimalTest().DoIt()` - Simple neural network
- `new OneNeuronTest().DoIt()` - Single neuron learning
- `new AdderTest().DoIt()` - Addition neural network
- `new Linear2LayersTest().DoIt()` - Linear two-layer network
- `new TwoLayerAdder().DoIt()` - Two-layer addition network
- `new Multiplier().DoIt()` - Multiplication network (currently active)

## Testing

This project includes a comprehensive test suite built with xUnit that covers all neural network components.

### Test Coverage
The project includes comprehensive test suites covering:

#### Core Components (`SimpleNeuralNetwork.Tests/`)
- **Activation Functions**: All activation functions and their derivatives
- **Neural Network**: Network construction, prediction, and validation
- **Training**: Backpropagation algorithms, weight updates, convergence
- **Layers & Neurons**: Individual component behavior and integration
- **Input Processing**: Data preprocessing and formatting
- **Loss Functions**: Error calculation and optimization metrics
- **Factory Patterns**: Component creation and initialization

#### Integration Tests
- **End-to-End Workflows**: Complete training and prediction scenarios
- **Complex Integration**: Multi-layer network testing
- **Error Handling**: Boundary conditions and error cases
- **Performance**: Gradient calculations and optimization

#### Specific Test Categories
- `ActivationFunctionsTests.cs`: Function behavior and derivatives
- `NeuralNetworkTests.cs`: Core network functionality
- `NeuralNetworkTrainerTests.cs`: Training algorithm validation
- `LayerTests.cs` & `NeuronTests.cs`: Component-level testing
- `InputProcessorTests.cs`: Data preprocessing validation
- `LossFunctionsTests.cs`: Error calculation verification
- `NetworkCreatorTests.cs`: Factory pattern testing
- Various integration test files for specific scenarios

### Running Tests
```bash
# Run all tests in the solution
dotnet test BackPropagation.sln

# Run tests with detailed output
dotnet test --verbosity normal

# Run specific test project
dotnet test SimpleNeuralNetwork.Tests/

# Run specific test class
dotnet test --filter "ClassName~ActivationFunctionsTests"

# Run tests in watch mode (for development)
dotnet watch test

# Run tests with code coverage
dotnet test --collect:"XPlat Code Coverage"
```

### Test Projects
- **SimpleNeuralNetwork.Tests**: Primary test suite with comprehensive coverage
- **BackPropagation.Tests**: Additional specialized tests

For detailed testing documentation, see the test README files in each test project directory.

## Requirements
- .NET 9.0 SDK or later
- Linux, Windows, or macOS

## License
This project is for educational purposes.
