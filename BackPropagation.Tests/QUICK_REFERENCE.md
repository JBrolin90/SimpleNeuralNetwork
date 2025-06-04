# Quick Test Reference

## Essential Commands

```bash
# Run all tests
dotnet test

# Run tests with verbose output  
dotnet test --verbosity normal

# Run tests in watch mode (auto-rerun on file changes)
dotnet watch test

# Run tests with code coverage
dotnet test --collect:"XPlat Code Coverage"
```

## Test Filtering Examples

```bash
# Run specific test class
dotnet test --filter "FullyQualifiedName~NodeFactoryTests"
dotnet test --filter "FullyQualifiedName~NeuralNetworkTests"

# Run tests by method name pattern
dotnet test --filter "DisplayName~Unit"
dotnet test --filter "TestName~Create"

# Run tests by category/trait (if implemented)
dotnet test --filter "Category=Unit"
```

## Expected Results

- **Total Tests**: ~45 test cases
- **Status**: Most tests should pass (some may fail if implementation has issues)
- **Duration**: < 1 second for full test suite

## Current Test Status

As of the last run:
- ✅ **43 tests passed** - Core functionality working
- ❌ **2 tests failed** - Known issues with null argument validation
- ⏱️ **Duration**: ~0.9 seconds

## Failing Tests (Expected)

1. `NodeFactoryTests.Create_WithNullLayer_ShouldThrowException`
   - Issue: NodeFactory doesn't validate null layer parameter
   
2. `NeuralNetworkTests.Constructor_WithNullFactory_ShouldThrowException`  
   - Issue: NeuralNetwork throws NullReferenceException instead of ArgumentNullException

These failures indicate areas where the implementation could be improved with better input validation.
