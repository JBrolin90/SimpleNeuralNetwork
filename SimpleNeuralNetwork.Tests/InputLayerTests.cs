using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class InputLayerTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void InputLayer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int index = 0;
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } }; // 2 nodes, 2 inputs each
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();

            // Act
            var inputLayer = new InputLayer(index, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(index, inputLayer.Index);
            Assert.Equal(2, inputLayer.Nodes.Length);
            Assert.Equal(weights, inputLayer.Weights);
            Assert.Equal(biases, inputLayer.Biases);
        }

        [Fact]
        public void InputLayer_Forward_PassesThroughInputsDirectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } }; // Identity matrix
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 2.5, 3.7 };

            // Act
            double[] outputs = inputLayer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(2.5, outputs[0], 7); // 2.5*1.0 + 3.7*0.0 + 0.0
            Assert.Equal(3.7, outputs[1], 7); // 2.5*0.0 + 3.7*1.0 + 0.0
        }

        [Fact]
        public void InputLayer_Forward_WithUnitActivation_ReturnsInputsPlusBias()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } };
            double[][] biases = { new double[] { 0.1 }, new double[] { 0.2 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 1.0, 2.0 };

            // Act
            double[] outputs = inputLayer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(1.1, outputs[0], 7); // 1.0*1.0 + 2.0*0.0 + 0.1
            Assert.Equal(2.2, outputs[1], 7); // 1.0*0.0 + 2.0*1.0 + 0.2
        }

        [Fact]
        public void InputLayer_PreviousLayer_ShouldAlwaysBeNull()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act & Assert
            Assert.Null(inputLayer.PreviousLayer);
        }

        [Fact]
        public void InputLayer_Backward_ShouldHandleBackpropagation()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            
            // Process some inputs first
            inputLayer.Forward(new double[] { 1.0 });
            
            var nodeSteps = new NodeSteps[] { new NodeSteps(1) };
            double error = 0.1;

            // Act
            var result = inputLayer.Backward(error, nodeSteps);

            // Assert
            Assert.Single(result);
            Assert.NotNull(result[0]);
        }

        [Theory]
        [InlineData(new double[] { 0.0 }, new double[] { 0.0 })]
        [InlineData(new double[] { 1.5 }, new double[] { 1.5 })]
        [InlineData(new double[] { -2.3 }, new double[] { -2.3 })]
        public void InputLayer_WithIdentityWeightsAndZeroBias_PassesThroughValues(double[] inputs, double[] expected)
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } }; // Single input, single output
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double[] outputs = inputLayer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(expected[0], outputs[0], 7);
        }

        [Fact]
        public void InputLayer_MultipleInputs_ProcessesEachIndependently()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 2.0, 0.0, 0.0 }, 
                new double[] { 0.0, 0.5, 0.0 }, 
                new double[] { 0.0, 0.0, -1.0 } 
            };
            double[][] biases = { 
                new double[] { 0.1 }, 
                new double[] { 0.0 }, 
                new double[] { 0.5 } 
            };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 1.0, 4.0, -2.0 };

            // Act
            double[] outputs = inputLayer.Forward(inputs);

            // Assert
            Assert.Equal(3, outputs.Length);
            Assert.Equal(2.1, outputs[0], 7); // 1.0*2.0 + 4.0*0.0 + (-2.0)*0.0 + 0.1
            Assert.Equal(2.0, outputs[1], 7); // 1.0*0.0 + 4.0*0.5 + (-2.0)*0.0 + 0.0
            Assert.Equal(2.5, outputs[2], 7); // 1.0*0.0 + 4.0*0.0 + (-2.0)*(-1.0) + 0.5
        }

        [Fact]
        public void InputLayer_GetWeightChainFactor_ReturnsCorrectValue()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double chainFactor = inputLayer.GetWeightChainFactor(0);

            // Assert
            Assert.Equal(1.0, chainFactor, 7); // Should return 1.0 for terminal layer
        }

        [Fact]
        public void InputLayer_GetBiasChainFactor_ReturnsCorrectValue()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var inputLayer = new InputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double chainFactor = inputLayer.GetBiasChainFactor();

            // Assert
            Assert.Equal(1.0, chainFactor, 7); // Should return 1.0 for terminal layer
        }
    }
}
