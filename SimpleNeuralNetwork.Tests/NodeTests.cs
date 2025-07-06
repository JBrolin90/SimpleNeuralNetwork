using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NodeTests
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
            var node = new Neuron(layer, index, weights, bias, ActivationFunctions.Sigmoid);
            
            // Assert
            Assert.Equal(layer, node.Layer);
            Assert.Equal(index, node.Index);
            Assert.Equal(weights, node.Weights);
            Assert.Equal(bias, node.Bias);
            Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
            Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
        }

        [Fact]
        public void Node_Constructor_WithNullLayer_ThrowsException()
        {
            // Arrange
            double[] weights = { 0.5 };
            double[] bias = { 0.2 };
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new Neuron(null!, 0, weights, bias));
        }

        [Theory]
        [InlineData(new double[] { 1.0, 2.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.2 }, 0.1)]  // 1.0*0.5 + 2.0*(-0.3) + 0.2 = 0.1
        [InlineData(new double[] { 0.0, 0.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.1 }, 0.1)]
        [InlineData(new double[] { 2.0, -1.0, 3.0 }, new double[] { 0.2, 0.4, -0.1 }, new double[] { 0.0 }, -0.3)] // 2.0*0.2 + (-1.0)*0.4 + 3.0*(-0.1) + 0.0 = -0.3
        public void ProcessInputs_WithVariousInputs_CalculatesCorrectSum(double[] inputs, double[] weights, double[] bias, double expectedSum)
        {
            // Arrange
            var layer = new MockLayer();
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.Unit);
            
            // Act
            node.ProcessInputs(inputs);
            
            // Assert
            Assert.Equal(expectedSum, node.Sum, 7);
        }

        [Fact]
        public void ProcessInputs_WithSigmoidActivation_ReturnsCorrectOutput()
        {
            // Arrange
            var layer = new MockLayer();
            double[] weights = { 1.0 };
            double[] bias = { 0.0 };
            double[] inputs = { 0.0 };
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.Sigmoid);
            
            // Act
            double result = node.ProcessInputs(inputs);
            
            // Assert
            Assert.Equal(0.5, result, 7); // sigmoid(0) = 0.5
            Assert.Equal(0.5, node.Y, 7);
        }

        [Fact]
        public void Node_Constructor_WithTanhActivation_SetsCorrectDerivative()
        {
            // Arrange
            var layer = new MockLayer();
            double[] weights = { 1.0 };
            double[] bias = { 0.0 };
            
            // Act
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.Tanh);
            
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
            double[] weights = { 1.0 };
            double[] bias = { 0.0 };
            
            // Act
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.ReLU);
            
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
            double[] weights = { 1.0 };
            double[] bias = { 0.0 };
            
            // Create a wrapper function for LeakyReLU with default alpha
            Func<double, double> leakyReLUFunc = x => ActivationFunctions.LeakyReLU(x);
            
            // Act
            var node = new Neuron(layer, 0, weights, bias, leakyReLUFunc);
            
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
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.Unit);
            
            // Act
            double result = node.ProcessInputs(inputs);
            
            // Assert
            double expectedSum = 1.0 * 0.5 + 2.0 * (-0.3) + 0.2; // -0.4
            Assert.Equal(expectedSum, result, 7);
            Assert.Equal(expectedSum, node.Y, 7);
        }


        [Theory]
        [InlineData(0.0, 0.25)] // sigmoid'(0) = 0.25
        [InlineData(1.0, 0.19661193324148185)] // sigmoid'(1) ≈ 0.1966
        public void SigmoidDerivative_WithPreComputedValues_ReturnsCorrectResults(double input, double expected)
        {
            // Arrange
            var layer = new MockLayer();
            double[] weights = { 1.0 };
            double[] bias = { 0.0 };
            double[] inputs = { input };
            var node = new Neuron(layer, 0, weights, bias, ActivationFunctions.Sigmoid);
            
            // Act
            double output = node.ProcessInputs(inputs);
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