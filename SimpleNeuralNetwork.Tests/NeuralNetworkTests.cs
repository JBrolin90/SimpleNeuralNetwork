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

        [Fact]
        public void NeuralNetwork_Predict_WithNullInput_ThrowsException()
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
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Act & Assert
            Assert.Throws<NullReferenceException>(() => neuralNetwork.Predict(null!));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithEmptyInput_WorksWithCurrentImplementation()
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
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Act - Current implementation may handle empty arrays differently
            var result = Record.Exception(() => neuralNetwork.Predict(new double[0]));

            // Assert - Document the current behavior
            // Note: This test documents the current behavior; in a real implementation
            // you might want to add proper validation
            Assert.Null(result); // Current implementation doesn't throw
        }

        [Theory]
        [InlineData(new int[] {2, 3, 1})]
        [InlineData(new int[] {1, 5, 3, 1})]
        [InlineData(new int[] {3, 10, 5, 2})]
        public void NeuralNetwork_WithDifferentArchitectures_WorksCorrectly(int[] layerSizes)
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            var weights = CreateSimpleWeights(layerSizes);
            var biases = CreateZeroBiases(layerSizes);
            var ys = CreateZeroYs(layerSizes);
            var activationFunctions = CreateUnitActivations(layerSizes.Length);
            
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            var inputs = new double[layerSizes[0]];
            for (int i = 0; i < inputs.Length; i++) inputs[i] = 1.0;

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Equal(layerSizes[^1], outputs.Length);
            Assert.Equal(layerSizes.Length, neuralNetwork.Layers.Length);
        }

        [Fact]
        public void NeuralNetwork_Predict_WithMultipleInputsAndOutputs_ReturnsCorrectDimensions()
        {
            // Arrange - Create a network that actually has multiple outputs
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { 
                    new double[] { 1.0, 0.5 }, // Neuron 1
                    new double[] { 0.5, 1.0 }  // Neuron 2
                }
            };
            double[][][] biases = {
                new double[][] { 
                    new double[] { 0.1 }, // Bias for neuron 1
                    new double[] { 0.2 }  // Bias for neuron 2
                }
            };
            double[][] ys = {
                new double[] { 0.0, 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 1.0, 2.0 };

            // Act
            double[] outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            // First neuron: 1*1+2*0.5+0.1 = 2.1
            // Second neuron: 1*0.5+2*1+0.2 = 2.7
            Assert.Equal(2.1, outputs[0], 7);
            Assert.Equal(2.7, outputs[1], 7);
        }

        [Fact]
        public void NeuralNetwork_Predict_WithVeryDeepNetwork_WorksCorrectly()
        {
            // Arrange - Create a 6-layer network: 1->2->3->2->1->1
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            var layerSizes = new int[] { 1, 2, 3, 2, 1, 1 };
            var weights = CreateSimpleWeights(layerSizes);
            var biases = CreateZeroBiases(layerSizes);
            var ys = CreateZeroYs(layerSizes);
            var activationFunctions = CreateUnitActivations(layerSizes.Length);
            
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 1.0 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.True(outputs[0] != 0.0); // Should produce some non-zero output
        }

        [Fact]
        public void NeuralNetwork_Predict_WithExtremeWeights_HandlesGracefully()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { double.MaxValue / 1000 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 1.0 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.False(double.IsNaN(outputs[0]));
            Assert.False(double.IsInfinity(outputs[0]));
        }

        [Theory]
        [InlineData(0.0)]
        [InlineData(1.0)]
        [InlineData(-1.0)]
        [InlineData(100.0)]
        [InlineData(-100.0)]
        public void NeuralNetwork_Predict_WithExtremeInputs_ReturnsValidOutputs(double extremeInput)
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
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { extremeInput };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(extremeInput, outputs[0], 7);
            Assert.False(double.IsNaN(outputs[0]));
            Assert.False(double.IsInfinity(outputs[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithSigmoidActivation_OutputsInValidRange()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 10.0 } }
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

            // Act & Assert for various inputs
            var testInputs = new double[] { -10.0, -1.0, 0.0, 1.0, 10.0 };
            foreach (var input in testInputs)
            {
                var outputs = neuralNetwork.Predict(new double[] { input });
                Assert.Single(outputs);
                Assert.True(outputs[0] >= 0.0 && outputs[0] <= 1.0, $"Sigmoid output {outputs[0]} should be between 0 and 1 for input {input}");
            }
        }

        [Fact]
        public void NeuralNetwork_Properties_AreReadOnlyAfterConstruction()
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
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Assert - Properties should reference the same arrays
            Assert.Same(weights, neuralNetwork.Weigths);
            Assert.Same(biases, neuralNetwork.Biases);
            Assert.Same(ys, neuralNetwork.Ys);
            Assert.Same(activationFunctions, neuralNetwork.ActivationFunctions);
        }

        [Fact]
        public void NeuralNetwork_LayerConnections_AreSetUpCorrectly()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layerSizes = new int[] { 2, 3, 1 };
            var weights = CreateSimpleWeights(layerSizes);
            var biases = CreateZeroBiases(layerSizes);
            var ys = CreateZeroYs(layerSizes);
            var activationFunctions = CreateUnitActivations(layerSizes.Length);

            // Act
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Assert
            for (int i = 0; i < neuralNetwork.Layers.Length; i++)
            {
                var layer = neuralNetwork.Layers[i];
                
                if (i == 0)
                {
                    Assert.Null(layer.PreviousLayer);
                }
                else
                {
                    Assert.Equal(neuralNetwork.Layers[i - 1], layer.PreviousLayer);
                }

                if (i == neuralNetwork.Layers.Length - 1)
                {
                    Assert.Null(layer.NextLayer);
                }
                else
                {
                    Assert.Equal(neuralNetwork.Layers[i + 1], layer.NextLayer);
                }
            }
        }

        [Fact]
        public void NeuralNetwork_MultiplePredictions_ProduceConsistentResults()
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
            double[] inputs = { 3.0 };

            // Act
            var output1 = neuralNetwork.Predict(inputs);
            var output2 = neuralNetwork.Predict(inputs);
            var output3 = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Equal(output1[0], output2[0], 7);
            Assert.Equal(output2[0], output3[0], 7);
            Assert.Equal(7.0, output1[0], 7); // 3.0 * 2.0 + 1.0 = 7.0
        }

        [Fact]
        public void NeuralNetwork_Predict_WithVeryLargeInputs_HandlesGracefully()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 0.001 } } // Small weights to avoid overflow
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid // Sigmoid naturally bounds output
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 1000000.0 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.True(outputs[0] >= 0.0 && outputs[0] <= 1.0);
            Assert.False(double.IsNaN(outputs[0]));
            Assert.False(double.IsInfinity(outputs[0]));
        }

        [Fact]
        public void NeuralNetwork_Predict_WithZeroWeights_ReturnsActivatedBias()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 0.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 5.0 } }
            };
            double[][] ys = {
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 123.456 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(5.0, outputs[0], 7); // Should be just the bias since weight is 0
        }

        [Fact]
        public void NeuralNetwork_Predict_WithComplexMultiLayerArchitecture_WorksCorrectly()
        {
            // Arrange - Create a 2-3-2-1 architecture
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            // Layer 1: 2 inputs -> 3 neurons
            // Layer 2: 3 inputs -> 2 neurons
            // Layer 3: 2 inputs -> 1 neuron
            double[][][] weights = {
                new double[][] { 
                    new double[] { 0.1, 0.2 }, // Neuron 1 of layer 1
                    new double[] { 0.3, 0.4 }, // Neuron 2 of layer 1
                    new double[] { 0.5, 0.6 }  // Neuron 3 of layer 1
                },
                new double[][] { 
                    new double[] { 0.1, 0.2, 0.3 }, // Neuron 1 of layer 2
                    new double[] { 0.4, 0.5, 0.6 }  // Neuron 2 of layer 2
                },
                new double[][] { 
                    new double[] { 0.7, 0.8 } // Neuron 1 of layer 3
                }
            };
            double[][][] biases = {
                new double[][] { 
                    new double[] { 0.1 }, // Bias for neuron 1
                    new double[] { 0.2 }, // Bias for neuron 2
                    new double[] { 0.3 }  // Bias for neuron 3
                },
                new double[][] { 
                    new double[] { 0.1 }, // Bias for neuron 1
                    new double[] { 0.2 }  // Bias for neuron 2
                },
                new double[][] { 
                    new double[] { 0.1 } // Bias for neuron 1
                }
            };
            double[][] ys = {
                new double[] { 0.0, 0.0, 0.0 },
                new double[] { 0.0, 0.0 },
                new double[] { 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 1.0, 2.0 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.True(outputs[0] > 0.0); // Should produce some positive output
            Assert.Equal(3, neuralNetwork.Layers.Length);
            Assert.Equal(3, neuralNetwork.Layers[0].Neurons.Length);
            Assert.Equal(2, neuralNetwork.Layers[1].Neurons.Length);
            Assert.Single(neuralNetwork.Layers[2].Neurons);
        }

        [Fact]
        public void NeuralNetwork_Predict_WithMixedActivationFunctions_WorksCorrectly()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.0 } }, // Unit activation
                new double[][] { new double[] { 1.0 } }  // Sigmoid activation
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
                ActivationFunctions.Sigmoid
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            double[] inputs = { 0.0 };

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.5, outputs[0], 7); // Unit(0) -> Sigmoid(0) = 0.5
        }

        [Fact]
        public void NeuralNetwork_Predict_ConsistentResultsAcrossMultipleCalls()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { new double[] { 1.5 } },
                new double[][] { new double[] { 2.0 } }
            };
            double[][][] biases = {
                new double[][] { new double[] { 0.5 } },
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
            double[] inputs = { 3.0 };

            // Act - Multiple predictions
            var results = new List<double[]>();
            for (int i = 0; i < 5; i++)
            {
                results.Add(neuralNetwork.Predict(inputs));
            }

            // Assert - All results should be identical
            for (int i = 1; i < results.Count; i++)
            {
                Assert.Equal(results[0].Length, results[i].Length);
                for (int j = 0; j < results[0].Length; j++)
                {
                    Assert.Equal(results[0][j], results[i][j], 7);
                }
            }
        }

        [Theory]
        [InlineData(new double[] { 0.0, 0.0 }, new double[] { 0.1, 0.2 })]
        [InlineData(new double[] { 1.0, 0.0 }, new double[] { 1.1, 0.2 })]
        [InlineData(new double[] { 0.0, 1.0 }, new double[] { 0.6, 1.2 })]
        [InlineData(new double[] { 1.0, 1.0 }, new double[] { 1.6, 1.2 })]
        public void NeuralNetwork_Predict_WithDifferentInputCombinations_ReturnsExpectedOutputs(double[] inputs, double[] expectedOutputs)
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][][] weights = {
                new double[][] { 
                    new double[] { 1.0, 0.5 }, // Neuron 1: w1=1.0, w2=0.5
                    new double[] { 0.0, 1.0 }  // Neuron 2: w1=0.0, w2=1.0
                }
            };
            double[][][] biases = {
                new double[][] { 
                    new double[] { 0.1 }, // Bias for neuron 1
                    new double[] { 0.2 }  // Bias for neuron 2
                }
            };
            double[][] ys = {
                new double[] { 0.0, 0.0 }
            };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);

            // Act
            var outputs = neuralNetwork.Predict(inputs);

            // Assert
            Assert.Equal(expectedOutputs.Length, outputs.Length);
            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                Assert.Equal(expectedOutputs[i], outputs[i], 7);
            }
        }

        [Fact]
        public void NeuralNetwork_Properties_AreImmutableReferences()
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
                ActivationFunctions.Unit
            };
            var neuralNetwork = new NeuralNetwork(layerFactory, nodeFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
            
            // Store original references
            var originalWeights = neuralNetwork.Weigths;
            var originalBiases = neuralNetwork.Biases;
            var originalYs = neuralNetwork.Ys;
            var originalActivationFunctions = neuralNetwork.ActivationFunctions;

            // Act - Predict to potentially modify internal state
            neuralNetwork.Predict(new double[] { 1.0 });

            // Assert - References should remain the same
            Assert.Same(originalWeights, neuralNetwork.Weigths);
            Assert.Same(originalBiases, neuralNetwork.Biases);
            Assert.Same(originalYs, neuralNetwork.Ys);
            Assert.Same(originalActivationFunctions, neuralNetwork.ActivationFunctions);
        }

        // Helper methods for creating test data
        private double[][][] CreateRandomWeights(int[] layerSizes)
        {
            var weights = new double[layerSizes.Length][][];
            for (int i = 0; i < layerSizes.Length; i++)
            {
                weights[i] = new double[layerSizes[i]][];
                int inputSize = i == 0 ? layerSizes[0] : layerSizes[i - 1];
                for (int j = 0; j < layerSizes[i]; j++)
                {
                    weights[i][j] = new double[inputSize];
                    for (int k = 0; k < inputSize; k++)
                    {
                        weights[i][j][k] = 1.0; // Use 1.0 for predictable results
                    }
                }
            }
            return weights;
        }

        private double[][][] CreateSimpleWeights(int[] layerSizes)
        {
            var weights = new double[layerSizes.Length][][];
            for (int i = 0; i < layerSizes.Length; i++)
            {
                weights[i] = new double[layerSizes[i]][];
                int inputSize = i == 0 ? layerSizes[0] : layerSizes[i - 1];
                for (int j = 0; j < layerSizes[i]; j++)
                {
                    weights[i][j] = new double[inputSize];
                    for (int k = 0; k < inputSize; k++)
                    {
                        weights[i][j][k] = 0.1; // Use small values to avoid extreme outputs
                    }
                }
            }
            return weights;
        }

        private double[][][] CreateZeroBiases(int[] layerSizes)
        {
            var biases = new double[layerSizes.Length][][];
            for (int i = 0; i < layerSizes.Length; i++)
            {
                biases[i] = new double[layerSizes[i]][];
                for (int j = 0; j < layerSizes[i]; j++)
                {
                    biases[i][j] = new double[1]; // Each neuron has one bias
                    biases[i][j][0] = 0.0;
                }
            }
            return biases;
        }

        private double[][] CreateZeroYs(int[] layerSizes)
        {
            var ys = new double[layerSizes.Length][];
            for (int i = 0; i < layerSizes.Length; i++)
            {
                ys[i] = new double[layerSizes[i]];
                for (int j = 0; j < layerSizes[i]; j++)
                {
                    ys[i][j] = 0.0;
                }
            }
            return ys;
        }

        private Func<double, double>[] CreateUnitActivations(int layerCount)
        {
            var activations = new Func<double, double>[layerCount];
            for (int i = 0; i < layerCount; i++)
            {
                activations[i] = ActivationFunctions.Unit;
            }
            return activations;
        }
    }
}
