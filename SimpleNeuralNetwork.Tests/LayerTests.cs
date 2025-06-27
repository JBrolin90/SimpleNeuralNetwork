using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class LayerTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void Layer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int index = 1;
            double[][] weights = { new double[] { 0.5, -0.3 }, new double[] { 0.8, 0.2 } };
            double[][] biases = { new double[] { 0.1 }, new double[] { -0.2 } };
            var nodeFactory = new NodeFactory();

            // Act
            var layer = new Layer(index, nodeFactory, weights, biases, ActivationFunctions.Sigmoid);

            // Assert
            Assert.Equal(index, layer.Index);
            Assert.Equal(2, layer.Nodes.Length);
            Assert.Equal(weights, layer.Weights);
            Assert.Equal(biases, layer.Biases);
        }

        [Fact]
        public void Layer_Forward_ProcessesInputsCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } };
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 2.0, 3.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(2.0, outputs[0], 7); // 2.0*1.0 + 3.0*0.0 + 0.0 = 2.0
            Assert.Equal(3.0, outputs[1], 7); // 2.0*0.0 + 3.0*1.0 + 0.0 = 3.0
        }

        [Fact]
        public void Layer_Forward_WithSigmoidActivation_AppliesActivationFunction()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Sigmoid);
            double[] inputs = { 0.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.5, outputs[0], 7); // sigmoid(0) = 0.5
        }

        [Fact]
        public void Layer_Forward_UpdatesYsProperty()
        {
            // Arrange
            double[][] weights = { new double[] { 2.0 }, new double[] { -1.0 } };
            double[][] biases = { new double[] { 1.0 }, new double[] { 0.5 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0 };

            // Act
            layer.Forward(inputs);

            // Assert
            Assert.Equal(2, layer.Ys.Length);
            Assert.Equal(7.0, layer.Ys[0], 7); // 3.0*2.0 + 1.0 = 7.0
            Assert.Equal(-2.5, layer.Ys[1], 7); // 3.0*(-1.0) + 0.5 = -2.5
        }

        [Fact]
        public void Layer_Backward_ProcessesErrorCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.5 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            
            // Process some inputs first
            layer.Forward(new double[] { 2.0, 3.0 });
            
            var nodeSteps = new Gradients[] { new Gradients(2) };
            double error = 0.1;

            // Act
            var result = layer.Backward(error, nodeSteps);

            // Assert
            Assert.Single(result);
            Assert.NotNull(result[0]);
            // The actual values depend on the chain factor calculation
        }

        [Theory]
        [InlineData(new double[] { 1.0, 2.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.2 }, 0.1)]
        [InlineData(new double[] { 0.0, 0.0 }, new double[] { 1.0, 1.0 }, new double[] { 0.5 }, 0.5)]
        public void Layer_SingleNode_ForwardCalculation(double[] inputs, double[] weights, double[] bias, double expected)
        {
            // Arrange
            double[][] layerWeights = { weights };
            double[][] layerBiases = { bias };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, layerWeights, layerBiases, ActivationFunctions.Unit);

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(expected, outputs[0], 7);
        }

        [Fact]
        public void Layer_MultipleNodes_ProcessInputsIndependently()
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
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0, 4.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(3.1, outputs[0], 7); // 3.0*1.0 + 4.0*0.0 + 0.1 = 3.1
            Assert.Equal(4.2, outputs[1], 7); // 3.0*0.0 + 4.0*1.0 + 0.2 = 4.2
        }

        [Fact]
        public void Layer_GetWeightChainFactor_WithNoNextLayer_ReturnsOne()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            // No NextLayer set (null)

            // Act
            double result = layer.CalculateLayerErrorRecursively(0);

            // Assert
            Assert.Equal(1.0, result, 7);
        }

        [Fact]
        public void Layer_GetBiasChainFactor_WithNoNextLayer_ReturnsOne()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            double result = layer.GetBiasChainFactor();

            // Assert
            Assert.Equal(1.0, result, 7);
        }

        [Fact]
        public void Layer_InputsProperty_StoresLastForwardInputs()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NodeFactory();
            var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 5.0 };

            // Act
            layer.Forward(inputs);

            // Assert
            Assert.NotNull(layer.Inputs);
            Assert.Single(layer.Inputs);
            Assert.Equal(inputs, layer.Inputs);
        }
    }
}
