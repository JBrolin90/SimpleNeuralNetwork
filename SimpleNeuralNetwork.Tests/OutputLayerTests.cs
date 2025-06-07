using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class OutputLayerTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void OutputLayer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int index = 2;
            double[][] weights = { new double[] { 0.5, -0.3 }, new double[] { 0.8, 0.2 } };
            double[][] biases = { new double[] { 0.1 }, new double[] { -0.2 } };
            var nodeFactory = new NodeFactory();

            // Act
            var outputLayer = new OutputLayer(index, nodeFactory, weights, biases, ActivationFunctions.Sigmoid);

            // Assert
            Assert.Equal(index, outputLayer.Index);
            Assert.Equal(2, outputLayer.Nodes.Length);
            Assert.Equal(weights, outputLayer.Weights);
            Assert.Equal(biases, outputLayer.Biases);
        }

        [Fact]
        public void OutputLayer_Forward_ProcessesInputsCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } };
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 2.0, 3.0 };

            // Act
            double[] outputs = outputLayer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(2.0, outputs[0], 7); // 2.0*1.0 + 3.0*0.0 + 0.0 = 2.0
            Assert.Equal(3.0, outputs[1], 7); // 2.0*0.0 + 3.0*1.0 + 0.0 = 3.0
        }

        [Fact]
        public void OutputLayer_Forward_WithSigmoidActivation_AppliesActivationFunction()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Sigmoid);
            double[] inputs = { 0.0 };

            // Act
            double[] outputs = outputLayer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.5, outputs[0], 7); // sigmoid(0) = 0.5
        }

        [Fact]
        public void OutputLayer_NextLayer_ShouldAlwaysBeNull()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act & Assert
            Assert.Null(outputLayer.NextLayer);
        }

        [Fact]
        public void OutputLayer_GetWeightChainFactor_ReturnsOne()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double chainFactor = outputLayer.GetWeightChainFactor(0);

            // Assert
            Assert.Equal(1.0, chainFactor, 7); // Output layer should return 1.0
        }

        [Fact]
        public void OutputLayer_GetBiasChainFactor_ReturnsOne()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double chainFactor = outputLayer.GetBiasChainFactor();

            // Assert
            Assert.Equal(1.0, chainFactor, 7); // Output layer should return 1.0
        }

        [Fact]
        public void OutputLayer_Backward_ProcessesErrorCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.5 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            
            // Process some inputs first
            outputLayer.Forward(new double[] { 2.0, 3.0 });
            
            var nodeSteps = new NodeSteps[] { new NodeSteps(2) };
            double error = 0.1;

            // Act
            var result = outputLayer.Backward(error, nodeSteps);

            // Assert
            Assert.Single(result);
            Assert.NotNull(result[0]);
        }

        [Theory]
        [InlineData(new double[] { 1.0, 2.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.2 }, 0.1)]
        [InlineData(new double[] { 0.0, 0.0 }, new double[] { 1.0, 1.0 }, new double[] { 0.5 }, 0.5)]
        public void OutputLayer_SingleNode_ForwardCalculation(double[] inputs, double[] weights, double[] bias, double expected)
        {
            // Arrange
            double[][] layerWeights = { weights };
            double[][] layerBiases = { bias };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, layerWeights, layerBiases, ActivationFunctions.Unit);

            // Act
            double[] outputs = outputLayer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(expected, outputs[0], 7);
        }

        [Fact]
        public void OutputLayer_MultipleNodes_ProcessInputsIndependently()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 1.0, 0.0 }, // First node: only uses first input
                new double[] { 0.0, 1.0 }  // Second node: only uses second input
            };
            double[][] biases = { 
                new double[] { 0.1 }, 
                new double[] { 0.2 } 
            };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0, 4.0 };

            // Act
            double[] outputs = outputLayer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(3.1, outputs[0], 7); // 3.0*1.0 + 4.0*0.0 + 0.1 = 3.1
            Assert.Equal(4.2, outputs[1], 7); // 3.0*0.0 + 4.0*1.0 + 0.2 = 4.2
        }

        [Fact]
        public void OutputLayer_WithSoftMaxActivation_ProducesValidProbabilities()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 1.0, 0.0 }, 
                new double[] { 0.0, 1.0 } 
            };
            double[][] biases = { 
                new double[] { 0.0 }, 
                new double[] { 0.0 } 
            };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Sigmoid);
            double[] inputs = { 1.0, 2.0 };

            // Act
            double[] outputs = outputLayer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.True(outputs[0] >= 0 && outputs[0] <= 1); // Sigmoid output should be between 0 and 1
            Assert.True(outputs[1] >= 0 && outputs[1] <= 1); // Sigmoid output should be between 0 and 1
        }

        [Fact]
        public void OutputLayer_InputsProperty_StoresLastForwardInputs()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var outputLayer = new OutputLayer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 5.0 };

            // Act
            outputLayer.Forward(inputs);

            // Assert
            Assert.NotNull(outputLayer.Inputs);
            Assert.Single(outputLayer.Inputs);
            Assert.Equal(inputs, outputLayer.Inputs);
        }
    }
}
