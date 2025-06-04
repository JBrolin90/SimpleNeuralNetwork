#!/bin/bash

# BackPropagation Neural Network - Test Runner Script
# This script demonstrates various ways to run the test suite

echo "=== BackPropagation Neural Network Test Suite ==="
echo ""

# Navigate to the project directory
cd "$(dirname "$0")"

echo "🔧 Building the solution..."
dotnet build
if [ $? -ne 0 ]; then
    echo "❌ Build failed. Please fix compilation errors first."
    exit 1
fi
echo "✅ Build successful!"
echo ""

echo "🧪 Running all tests..."
dotnet test --nologo
echo ""

echo "📊 Test Summary:"
echo "- Running tests with detailed output..."
dotnet test --nologo --verbosity quiet | grep -E "(Passed|Failed|Skipped|Total)"
echo ""

echo "🔍 Available test classes:"
echo "1. ActivationFunctionTests (UnitTest1.cs) - Tests activation functions"
echo "2. NeuralNetworkTests - Tests basic neural network functionality" 
echo "3. TestNeuralNetworkTests - Tests training functionality"
echo "4. FactoryTests - Tests factory pattern implementations"
echo "5. NodeStepsTests - Tests gradient calculation data structures"
echo "6. LayerTests - Tests layer functionality"
echo "7. NodeTests - Tests node functionality"
echo "8. IntegrationTests - Tests end-to-end scenarios"
echo ""

echo "💡 Example commands to run specific test groups:"
echo ""
echo "# Run activation function tests:"
echo "dotnet test --filter \"DisplayName~Unit\""
echo ""
echo "# Run neural network tests:"
echo "dotnet test --filter \"FullyQualifiedName~NeuralNetworkTests\""
echo ""
echo "# Run factory tests:"
echo "dotnet test --filter \"FullyQualifiedName~FactoryTests\""
echo ""
echo "# Run in watch mode for development:"
echo "dotnet watch test"
echo ""

echo "📚 For detailed testing instructions, see:"
echo "- BackPropagation.Tests/README.md"
echo "- Main README.md (Testing section)"
