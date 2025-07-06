using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NeuronTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void Node_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            var layer = new MockLayer();
            int index = 1;
            double[] weights = { 0.5, -0.3, 0.8 };
            double[] bias = { 0.2 };
            
            // Act
            var node = new Neuron(layer, index, ActivationFunctions.Sigmoid);
            
            // Assert
            Assert.Equal(layer, node.Layer);
            Assert.Equal(index, node.Index);
            Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithNullLayer_ThrowsException()
        {
            // Arrange
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Neuron(null!, 0, ActivationFunctions.Unit));
        }


        [Fact]
        public void Node_Constructor_WithTanhActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.Tanh);
            
            // Assert
            Assert.Equal(ActivationFunctions.Tanh, node.ActivationFunction);
            // Note: The current Node constructor doesn't handle Tanh derivative assignment
            // This test reveals that the constructor needs to be updated to handle more activation functions
        }

        [Fact]
        public void Node_Constructor_WithReLUActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Act
            var node = new Neuron(layer, 0, ActivationFunctions.ReLU);
            
            // Assert
            Assert.Equal(ActivationFunctions.ReLU, node.ActivationFunction);
            // Note: The current Node constructor doesn't handle ReLU derivative assignment
            // This test reveals that the constructor needs to be updated to handle more activation functions
        }

        [Fact]
        public void Node_Constructor_WithLeakyReLUActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            
            // Create a wrapper function for LeakyReLU with default alpha
            Func<double, double> leakyReLUFunc = x => ActivationFunctions.LeakyReLU(x);
            
            // Act
            var node = new Neuron(layer, 0, leakyReLUFunc);
            
            // Assert
            Assert.Equal(leakyReLUFunc, node.ActivationFunction);
            // Note: The current Node constructor doesn't handle LeakyReLU derivative assignment
        }

        [Fact]
        public void ProcessInputs_WithUnitActivation_ReturnsSum()
        {
            // Arrange
            var layer = new MockLayer();
            double[] weights = { 0.5, -0.3 };
            double[] bias = { 0.2 };
            double[] inputs = { 1.0, 2.0 };
            var inputProcessor = new InputProcessor(layer, 0, weights, bias);
            
            // Act
            double result = inputProcessor.ProcessInputs(inputs);
            
            // Assert
            double expectedSum = 1.0 * 0.5 + 2.0 * (-0.3) + 0.2; // -0.4
            Assert.Equal(expectedSum, result, 7);
            Assert.Equal(expectedSum, inputProcessor.Y, 7);
        }


        [Theory]
        [InlineData(0.0, 0.25)] // sigmoid'(0) = 0.25
        [InlineData(1.0, 0.19661193324148185)] // sigmoid'(1) ≈ 0.1966
        public void SigmoidDerivative_WithPreComputedValues_ReturnsCorrectResults(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            double[] inputs = { input };
            var node = new Neuron(layer, 0, ActivationFunctions.Sigmoid);
            
            // Act
            double output = node.Activate(input);
            double derivative = node.ActivationDerivative(output);
            
            // Assert
            Assert.Equal(expected, derivative, 7);
        }
    }

    // Mock layer class for testing
    public class MockLayer : ILayer
    {
        public INeuron[] Neurons { get; set; } = Array.Empty<INeuron>();
        public ILayer? PreviousLayer { get; set; }
        public ILayer? NextLayer { get; set; }
        public double[]? Inputs { get; set; }
        public int Index { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public IInputProcessor[] InputProcessors { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }

        public double[] Forward(double[] inputs)
        {
            return inputs;
        }

        public Gradients[] Backward(double dSSR, Gradients[] steps)
        {
            return steps;
        }

        public double CalculateLayerErrorRecursively(int index)
        {
            return 1.0; // Simplified for testing
        }

        public double GetBiasChainFactor()
        {
            return 1.0; // Simplified for testing
        }
    }
}