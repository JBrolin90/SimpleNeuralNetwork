using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NeuralNetworkTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void NeuralNetwork_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0, 0.0 } }, // Input layer
                new double[][] { new double[] { 0.5, 0.5 } }  // Output layer
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.1 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Sigmoid
            };

            // Act
            var neuralNetwork = new NeuralNetwork(layerFactory, neuronFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Assert
            Assert.Equal(2, neuralNetwork.Layers.Length);
            Assert.Equal(weights, neuralNetwork.Weigths);
            Assert.Equal(biases, neuralNetwork.Biases);
            Assert.Equal(ys, neuralNetwork.Ys);
            Assert.Equal(activationFunctions, neuralNetwork.ActivationFunctions);
        }

        [Fact]
        public void NeuralNetwork_Constructor_WithNullLayerFactory_ThrowsException()
        {
            // Arrange
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = { new double[][] { new double[] { 1.0 } } };
            double[][][] biases = { new double[][] { new double[] { 0.0 } } };
            double[][] ys = { new double[] { 0.0 } };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                new NeuralNetwork(null!, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions));
        }

        [Fact]
        public void NeuralNetwork_Constructor_WithNullNodeFactory_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = { new double[][] { new double[] { 1.0 } } };
            double[][][] biases = { new double[][] { new double[] { 0.0 } } };
            double[][] ys = { new double[] { 0.0 } };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                new NeuralNetwork(layerFactory, null!, inputProcessorFactory, weights, biases, ys, activationFunctions));
        }

        [Fact]
        public void NeuralNetwork_Constructor_SetsUpLayerConnections()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }, // Input layer
                new double[][] { new double[] { 1.0 } }, // Hidden layer
                new double[][] { new double[] { 1.0 } }  // Output layer
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 },
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };

            // Act
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Assert
            Assert.Equal(3, neuralNetwork.Layers.Length);
            
            // Check layer connections
            Assert.Null(neuralNetwork.Layers[0].PreviousLayer); // Input layer has no previous
            Assert.Equal(neuralNetwork.Layers[1], neuralNetwork.Layers[0].NextLayer);
            
            Assert.Equal(neuralNetwork.Layers[0], neuralNetwork.Layers[1].PreviousLayer);
            Assert.Equal(neuralNetwork.Layers[2], neuralNetwork.Layers[1].NextLayer);
            
            Assert.Equal(neuralNetwork.Layers[1], neuralNetwork.Layers[2].PreviousLayer);
            Assert.Null(neuralNetwork.Layers[2].NextLayer); // Output layer has no next
        }

        [Fact]
        public void NeuralNetwork_Predict_WithSingleLayer_ReturnsCorrectOutput()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 2.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.5 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 3.0 };

            // Act
            double[] outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(6.5, outputs[0], 7); // 3.0 * 2.0 + 0.5 = 6.5
        }

        [Fact]
        public void NeuralNetwork_Predict_WithMultipleLayers_PropagatesForward()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }, // Input: pass through
                new double[][] { new double[] { 2.0 } }  // Output: multiply by 2
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 1.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 5.0 };

            // Act
            double[] outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(11.0, outputs[0], 7); // First layer: 5.0, Second layer: 5.0 * 2.0 + 1.0 = 11.0
        }

        [Fact]
        public void NeuralNetwork_Predict_UpdatesYsProperty()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } },
                new double[][] { new double[] { 3.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } },
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 },
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 2.0 };

            // Act
            neuralNetwork.Predict(inputs);

            // Assert
            Assert.Equal(2.0, neuralNetwork.Ys[0][0], 7); // First layer output: 2.0
            Assert.Equal(6.0, neuralNetwork.Ys[1][0], 7); // Second layer output: 2.0 * 3.0 = 6.0
        }

        [Theory]
        [InlineData(new double[] { 1.0 }, 3.0)]
        [InlineData(new double[] { 0.0 }, 1.0)]
        [InlineData(new double[] { -1.0 }, -1.0)]
        public void NeuralNetwork_Predict_WithVariousInputs_ReturnsExpectedOutputs(double[] inputs, double expected)
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 2.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Act
            double[] outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(expected, outputs[0], 7);
        }

        [Fact]
        public void NeuralNetwork_Predict_WithSigmoidActivation_AppliesActivationFunction()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 0.0 };

            // Act
            double[] outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.5, outputs[0], 7); // sigmoid(0) = 0.5
        }
    }
}
