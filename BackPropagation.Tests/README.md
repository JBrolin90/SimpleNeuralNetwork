# BackPropagation Neural Network - Test Suite

This directory contains comprehensive unit tests for the BackPropagation neural network implementation using xUnit testing framework.

## Test Coverage

The test suite provides comprehensive coverage for all neural network components:

### Core Components
- **ActivationFunctionTests** (`UnitTest1.cs`) - Tests for all activation functions (Unit, SoftPlus, Sigmoid, ReLU) and their derivatives
- **NeuralNetworkTests** (`NeuralNetworkTests.cs`) - Tests for the main NeuralNetwork class constructor and prediction methods
- **TestNeuralNetworkTests** (`TestNeuralNetworkTests.cs`) - Tests for training functionality and weight updates
- **LayerTests** (`LayerTests.cs`) - Tests for Layer class forward propagation and connectivity
- **NodeTests** (`NodeTests.cs`) - Tests for Node class input processing and activation

### Factory Pattern
- **FactoryTests** (`FactoryTests.cs`) - Tests for NodeFactory and LayerFactory classes

### Data Structures
- **NodeStepsTests** (`NodeStepsTests.cs`) - Tests for NodeSteps data structure used in backpropagation

### Integration
- **IntegrationTests** (`IntegrationTests.cs`) - End-to-end integration tests for complete neural network workflows

## Running Tests

### Prerequisites
- .NET 9.0 SDK or later
- The solution should be built successfully

### Command Line Options

#### Run All Tests
```bash
# From the solution root directory
dotnet test

# Or from the test project directory
cd BackPropagation.Tests
dotnet test
```

#### Run Tests with Detailed Output
```bash
dotnet test --verbosity normal
```

#### Run Tests with Coverage
```bash
dotnet test --collect:"XPlat Code Coverage"
```

#### Run Specific Test Class
```bash
# Run only activation function tests
dotnet test --filter "FullyQualifiedName~ActivationFunctionTests"

# Run only neural network tests
dotnet test --filter "FullyQualifiedName~NeuralNetworkTests"

# Run only factory tests
dotnet test --filter "FullyQualifiedName~FactoryTests"
```

#### Run Specific Test Method
```bash
# Run a specific test method
dotnet test --filter "FullyQualifiedName~ActivationFunctionTests.TestUnitFunction"
```

#### Run Tests in Watch Mode (for development)
```bash
dotnet watch test
```

### Visual Studio / VS Code

#### Visual Studio
1. Open the solution file `BackPropagation.sln`
2. Build the solution (Build → Build Solution)
3. Open Test Explorer (Test → Test Explorer)
4. Click "Run All Tests" or run individual tests

#### VS Code
1. Install the C# extension and .NET Test Explorer extension
2. Open the workspace folder
3. The Test Explorer will automatically discover tests
4. Click the test icons in the gutter or use the Test Explorer panel

### Expected Test Results

All tests should pass when the neural network implementation is correct. The test suite includes:

- **57+ individual test cases** covering various scenarios
- **Input validation tests** to ensure proper error handling
- **Boundary condition tests** for edge cases
- **Integration tests** for end-to-end functionality

### Test Structure

Each test file follows the Arrange-Act-Assert pattern:

```csharp
[Fact]
public void TestMethod_Scenario_ExpectedResult()
{
    // Arrange - Set up test data and dependencies
    var factory = new NodeFactory();
    
    // Act - Execute the method being tested
    var result = factory.Create(parameters);
    
    // Assert - Verify the expected outcome
    Assert.NotNull(result);
    Assert.Equal(expectedValue, result.Property);
}
```

### Troubleshooting

#### Common Issues

1. **Build Fails Before Tests Run**
   ```bash
   dotnet clean
   dotnet restore
   dotnet build
   ```

2. **Tests Not Discovered**
   - Ensure the test project has proper xUnit references
   - Check that test methods are marked with `[Fact]` or `[Theory]` attributes
   - Verify the test class is public

3. **Reference Errors**
   - Ensure the test project references the main project
   - Check that all required using statements are present

#### Debugging Tests

To debug a failing test:

1. Set breakpoints in your test code
2. Run tests in debug mode:
   ```bash
   dotnet test --logger "console;verbosity=detailed"
   ```

3. Or use your IDE's debugging capabilities

### Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Include both positive and negative test cases
3. Use descriptive test method names that explain the scenario
4. Add appropriate test data using `[Theory]` and `[InlineData]` for parameterized tests
5. Ensure tests are independent and don't rely on execution order

### Performance

The test suite is designed to run quickly:
- Most tests complete in milliseconds
- No external dependencies or file I/O
- Lightweight test data and minimal setup

For performance testing of the neural network itself, see the integration tests which include timing assertions for training operations.
