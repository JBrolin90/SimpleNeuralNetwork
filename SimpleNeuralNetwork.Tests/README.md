# SimpleNeuralNetwork.Tests

This project contains comprehensive unit tests for the SimpleNeuralNetwork library using xUnit testing framework.

## Overview

The test suite provides comprehensive coverage for all major components of the neural network library, including:

- **Activation Functions** - Testing mathematical functions and their derivatives
- **Node Operations** - Testing individual neuron functionality 
- **Layer Operations** - Testing layer-level forward and backward propagation
- **Network Architecture** - Testing complete neural network functionality
- **Training Components** - Testing backpropagation and weight updates
- **Network Creation** - Testing network initialization and configuration

## Test Structure

### Core Component Tests

#### ActivationFunctionTests
- Tests all activation functions (Sigmoid, ReLU, Tanh, LeakyReLU, Unit, SoftPlus)
- Validates function outputs with known mathematical values
- Tests derivative calculations
- Verifies edge cases and boundary conditions

#### NodeTests
- Tests node construction and initialization
- Validates forward propagation through individual nodes
- Tests backpropagation calculations
- Verifies weight and bias derivative computations

#### LayerTests
- Tests layer construction with multiple nodes
- Validates forward pass through layers
- Tests backward propagation through layers
- Verifies layer connectivity and chain factor calculations

#### InputLayerTests
- Tests input layer specific functionality
- Validates input processing and pass-through behavior
- Tests layer properties and constraints

#### OutputLayerTests
- Tests output layer specific functionality
- Validates final output generation
- Tests terminal layer behavior

### Network-Level Tests

#### NeuralNetworkTests
- Tests complete network construction
- Validates end-to-end forward propagation
- Tests network property initialization
- Verifies multi-layer connectivity

#### NeuralNetworkTrainerTests
- Tests training loop functionality
- Validates backpropagation implementation
- Tests weight and bias updates
- Verifies learning rate effects
- Tests error calculation and accumulation

#### NetworkCreatorTests
- Tests network architecture creation
- Validates weight and bias initialization
- Tests random weight generation
- Verifies network dimension calculations

### Utility Tests

#### NodeStepsTests
- Tests gradient step accumulation
- Validates weight step arrays
- Tests bias step calculations

## Running the Tests

### Prerequisites
- .NET 9.0 SDK
- xUnit test runner

### Command Line
```bash
# Run all tests
dotnet test

# Run tests with detailed output
dotnet test --verbosity normal

# Run specific test class
dotnet test --filter "ClassName=NodeTests"

# Run specific test method
dotnet test --filter "MethodName=Node_Constructor_InitializesPropertiesCorrectly"
```

### Visual Studio
1. Open the solution in Visual Studio
2. Build the solution (Ctrl+Shift+B)
3. Open Test Explorer (Test → Test Explorer)
4. Run all tests or select specific tests

### Coverage
The test suite provides comprehensive coverage including:
- Happy path scenarios
- Edge cases and boundary conditions
- Error conditions and exception handling
- Mathematical correctness validation
- Integration between components

## Test Categories

### Unit Tests
Most tests are isolated unit tests that test individual components in isolation using mock objects where necessary.

### Integration Tests
Some tests verify integration between components (e.g., full network forward/backward passes).

### Mathematical Validation
Special focus on validating mathematical correctness of:
- Activation function calculations
- Derivative computations
- Gradient calculations
- Weight update formulas

## Test Data and Patterns

### Theory Tests
Many tests use xUnit's `[Theory]` attribute to test multiple input scenarios:
```csharp
[Theory]
[InlineData(input1, expected1)]
[InlineData(input2, expected2)]
public void TestMethod(double input, double expected)
```

### Precision Testing
All floating-point comparisons use appropriate tolerance values:
```csharp
Assert.Equal(expected, actual, 7); // 7 decimal places precision
```

### Mock Objects
Tests use mock implementations (e.g., `MockLayer`) to isolate components under test.

## Contributing

When adding new tests:

1. Follow the existing naming convention: `ClassName_MethodName_ExpectedBehavior`
2. Use appropriate test categories (Fact, Theory)
3. Include comprehensive arrange/act/assert sections
4. Add appropriate comments explaining complex test scenarios
5. Use meaningful test data that covers edge cases
6. Maintain consistent precision tolerances for floating-point comparisons

## Dependencies

- **xUnit** (2.9.3) - Testing framework
- **Microsoft.NET.Test.Sdk** (17.11.1) - Test SDK
- **xunit.runner.visualstudio** (2.8.2) - Visual Studio test runner
- **coverlet.collector** (6.0.2) - Code coverage collection

## Notes

- All tests target .NET 9.0
- Nullable reference types are enabled
- Tests are designed to be deterministic and repeatable
- Random number generation uses fixed seeds where deterministic behavior is required
