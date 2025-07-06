using Xunit;
using System;
using System.Linq;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class LayerTests
    {
        private const double Tolerance = 1e-7;

        #region Constructor Tests

        [Fact]
        public void Layer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int index = 1;
            double[][] weights = { new double[] { 0.5, -0.3 }, new double[] { 0.8, 0.2 } };
            double[][] biases = { new double[] { 0.1 }, new double[] { -0.2 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            var layer = new Layer(index, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);

            // Assert
            Assert.Equal(index, layer.Index);
            Assert.Equal(2, layer.Neurons.Length);
            Assert.Equal(2, layer.InputProcessors.Length);
            Assert.Equal(2, layer.Ys.Length);
            Assert.Equal(weights, layer.Weights);
            Assert.Equal(biases, layer.Biases);
            Assert.Null(layer.Inputs);
            Assert.Null(layer.PreviousLayer);
            Assert.Null(layer.NextLayer);
        }

        [Fact]
        public void Layer_Constructor_CreatesCorrectNumberOfNeurons()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 1.0, 2.0 }, 
                new double[] { 3.0, 4.0 }, 
                new double[] { 5.0, 6.0 } 
            };
            double[][] biases = { 
                new double[] { 0.1 }, 
                new double[] { 0.2 }, 
                new double[] { 0.3 } 
            };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(3, layer.Neurons.Length);
            Assert.Equal(3, layer.InputProcessors.Length);
            Assert.Equal(3, layer.Ys.Length);
            
            // Check that each neuron is properly initialized
            for (int i = 0; i < layer.Neurons.Length; i++)
            {
                Assert.NotNull(layer.Neurons[i]);
                Assert.Equal(i, layer.Neurons[i].Index);
                Assert.Equal(layer, ((Neuron)layer.Neurons[i]).Layer);
            }
        }

        [Fact]
        public void Layer_Constructor_CreatesCorrectInputProcessors()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] biases = { new double[] { 0.5 }, new double[] { -0.5 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(2, layer.InputProcessors.Length);
            
            // Check that each input processor is properly initialized
            for (int i = 0; i < layer.InputProcessors.Length; i++)
            {
                Assert.NotNull(layer.InputProcessors[i]);
                Assert.Equal(i, layer.InputProcessors[i].Index);
                Assert.Equal(layer, layer.InputProcessors[i].Layer);
                Assert.Equal(weights[i], layer.InputProcessors[i].Weights);
                Assert.Equal(biases[i], layer.InputProcessors[i].Bias);
            }
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(5)]
        [InlineData(10)]
        public void Layer_Constructor_WithDifferentIndices_SetsIndexCorrectly(int index)
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            var layer = new Layer(index, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(index, layer.Index);
        }

        [Fact]
        public void Layer_Constructor_WithDifferentActivationFunctions_InitializesCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act & Assert - Sigmoid
            var sigmoidLayer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);
            Assert.Equal(ActivationFunctions.Sigmoid, sigmoidLayer.Neurons[0].ActivationFunction);

            // Act & Assert - ReLU
            var reluLayer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.ReLU);
            Assert.Equal(ActivationFunctions.ReLU, reluLayer.Neurons[0].ActivationFunction);

            // Act & Assert - Tanh
            var tanhLayer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Tanh);
            Assert.Equal(ActivationFunctions.Tanh, tanhLayer.Neurons[0].ActivationFunction);
        }

        [Fact]
        public void Layer_Constructor_WithEmptyWeightsAndBiases_CreatesEmptyLayer()
        {
            // Arrange
            double[][] weights = Array.Empty<double[]>();
            double[][] biases = Array.Empty<double[]>();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Empty(layer.Neurons);
            Assert.Empty(layer.InputProcessors);
            Assert.Empty(layer.Ys);
        }

        #endregion

        #region Forward Method Tests

        [Fact]
        public void Layer_Forward_ProcessesInputsCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 } };
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 2.0, 3.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(2.0, outputs[0], Tolerance); // 2.0*1.0 + 3.0*0.0 + 0.0 = 2.0
            Assert.Equal(3.0, outputs[1], Tolerance); // 2.0*0.0 + 3.0*1.0 + 0.0 = 3.0
        }

        [Fact]
        public void Layer_Forward_WithSigmoidActivation_AppliesActivationFunction()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);
            double[] inputs = { 0.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.5, outputs[0], Tolerance); // sigmoid(0) = 0.5
        }

        [Fact]
        public void Layer_Forward_UpdatesYsProperty()
        {
            // Arrange
            double[][] weights = { new double[] { 2.0 }, new double[] { -1.0 } };
            double[][] biases = { new double[] { 1.0 }, new double[] { 0.5 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0 };

            // Act
            layer.Forward(inputs);

            // Assert
            Assert.Equal(2, layer.Ys.Length);
            Assert.Equal(7.0, layer.Ys[0], Tolerance); // 3.0*2.0 + 1.0 = 7.0
            Assert.Equal(-2.5, layer.Ys[1], Tolerance); // 3.0*(-1.0) + 0.5 = -2.5
        }

        [Theory]
        [InlineData(new double[] { 1.0, 2.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.2 }, 0.1)]
        [InlineData(new double[] { 0.0, 0.0 }, new double[] { 1.0, 1.0 }, new double[] { 0.5 }, 0.5)]
        [InlineData(new double[] { 5.0, -2.0 }, new double[] { 0.2, 0.8 }, new double[] { -0.1 }, -0.7)]
        public void Layer_SingleNode_ForwardCalculation(double[] inputs, double[] weights, double[] bias, double expected)
        {
            // Arrange
            double[][] layerWeights = { weights };
            double[][] layerBiases = { bias };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, layerWeights, layerBiases, ActivationFunctions.Unit);

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(expected, outputs[0], Tolerance);
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
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0, 4.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(3.1, outputs[0], Tolerance); // 3.0*1.0 + 4.0*0.0 + 0.1 = 3.1
            Assert.Equal(4.2, outputs[1], Tolerance); // 3.0*0.0 + 4.0*1.0 + 0.2 = 4.2
        }

        [Fact]
        public void Layer_Forward_WithReLUActivation_AppliesCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 }, new double[] { -1.0 } };
            double[][] biases = { new double[] { -1.0 }, new double[] { 1.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.ReLU);
            double[] inputs = { 2.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(1.0, outputs[0], Tolerance); // ReLU(2.0*1.0 + (-1.0)) = ReLU(1.0) = 1.0
            Assert.Equal(0.0, outputs[1], Tolerance); // ReLU(2.0*(-1.0) + 1.0) = ReLU(-1.0) = 0.0
        }

        [Fact]
        public void Layer_Forward_WithTanhActivation_AppliesCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Tanh);
            double[] inputs = { 0.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(0.0, outputs[0], Tolerance); // tanh(0) = 0.0
        }

        [Fact]
        public void Layer_Forward_MultipleCalls_UpdatesStateCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Act & Assert - First call
            double[] inputs1 = { 2.0 };
            double[] outputs1 = layer.Forward(inputs1);
            Assert.Equal(2.0, outputs1[0], Tolerance);
            Assert.Equal(inputs1, layer.Inputs);
            Assert.Equal(2.0, layer.Ys[0], Tolerance);

            // Act & Assert - Second call
            double[] inputs2 = { 5.0 };
            double[] outputs2 = layer.Forward(inputs2);
            Assert.Equal(5.0, outputs2[0], Tolerance);
            Assert.Equal(inputs2, layer.Inputs);
            Assert.Equal(5.0, layer.Ys[0], Tolerance);
        }

        [Fact]
        public void Layer_Forward_WithLargeInputs_ProcessesCorrectly()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 0.1, 0.2, 0.3, 0.4, 0.5 },
                new double[] { 0.5, 0.4, 0.3, 0.2, 0.1 }
            };
            double[][] biases = { new double[] { 0.0 }, new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 1.0, 2.0, 3.0, 4.0, 5.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            double expected1 = 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 4.0*0.4 + 5.0*0.5; // 5.5
            double expected2 = 1.0*0.5 + 2.0*0.4 + 3.0*0.3 + 4.0*0.2 + 5.0*0.1; // 3.5
            Assert.Equal(expected1, outputs[0], Tolerance);
            Assert.Equal(expected2, outputs[1], Tolerance);
        }

        #endregion

        #region Property Tests

        [Fact]
        public void Layer_InputsProperty_StoresLastForwardInputs()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 5.0 };

            // Act
            layer.Forward(inputs);

            // Assert
            Assert.NotNull(layer.Inputs);
            Assert.Single(layer.Inputs);
            Assert.Equal(inputs, layer.Inputs);
        }

        [Fact]
        public void Layer_PreviousLayer_CanBeSetAndRetrieved()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer1 = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            var layer2 = new Layer(1, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            layer2.PreviousLayer = layer1;

            // Assert
            Assert.Equal(layer1, layer2.PreviousLayer);
            Assert.Null(layer1.PreviousLayer);
        }

        [Fact]
        public void Layer_NextLayer_CanBeSetAndRetrieved()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer1 = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            var layer2 = new Layer(1, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            layer1.NextLayer = layer2;

            // Assert
            Assert.Equal(layer2, layer1.NextLayer);
            Assert.Null(layer2.NextLayer);
        }

        [Fact]
        public void Layer_Index_CanBeModified()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            layer.Index = 5;

            // Assert
            Assert.Equal(5, layer.Index);
        }

        [Fact]
        public void Layer_Neurons_CanBeModified()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            var newNeurons = new INeuron[] { new Neuron(layer, 0, ActivationFunctions.Sigmoid) };

            // Act
            layer.Neurons = newNeurons;

            // Assert
            Assert.Equal(newNeurons, layer.Neurons);
            Assert.Single(layer.Neurons);
        }

        [Fact]
        public void Layer_InputProcessors_CanBeModified()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            var newProcessors = new IInputProcessor[] { new InputProcessor(layer, 0, new double[] { 2.0 }, new double[] { 1.0 }) };

            // Act
            layer.InputProcessors = newProcessors;

            // Assert
            Assert.Equal(newProcessors, layer.InputProcessors);
            Assert.Single(layer.InputProcessors);
        }

        #endregion

        #region Interface Implementation Tests

        [Fact]
        public void Layer_ImplementsILayer()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();

            // Act
            ILayer layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.IsAssignableFrom<ILayer>(layer);
            Assert.Equal(0, layer.Index);
            Assert.NotNull(layer.Neurons);
            Assert.NotNull(layer.InputProcessors);
            Assert.Null(layer.PreviousLayer);
            Assert.Null(layer.NextLayer);
            Assert.Null(layer.Inputs);
        }

        [Fact]
        public void Layer_InterfaceMethods_WorkCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 2.0 } };
            double[][] biases = { new double[] { 1.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            ILayer layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 3.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(7.0, outputs[0], Tolerance); // 3.0*2.0 + 1.0 = 7.0
            Assert.Equal(inputs, layer.Inputs);
        }

        [Fact]
        public void Layer_InterfaceProperties_CanBeModified()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            ILayer layer1 = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            ILayer layer2 = new Layer(1, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Act
            layer1.Index = 10;
            layer1.NextLayer = layer2;
            layer2.PreviousLayer = layer1;

            // Assert
            Assert.Equal(10, layer1.Index);
            Assert.Equal(layer2, layer1.NextLayer);
            Assert.Equal(layer1, layer2.PreviousLayer);
        }

        #endregion

        #region Factory Tests

        [Fact]
        public void LayerFactory_Create_ReturnsLayerInstance()
        {
            // Arrange
            var factory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act
            var layer = factory.Create(3, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);

            // Assert
            Assert.NotNull(layer);
            Assert.IsType<Layer>(layer);
            Assert.Equal(3, layer.Index);
            Assert.Single(layer.Neurons);
            Assert.Single(layer.InputProcessors);
        }

        [Fact]
        public void LayerFactory_Create_WithDifferentActivationFunctions_CreatesCorrectly()
        {
            // Arrange
            var factory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };

            // Act & Assert - Unit
            var unitLayer = factory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            Assert.Equal(ActivationFunctions.Unit, unitLayer.Neurons[0].ActivationFunction);

            // Act & Assert - ReLU
            var reluLayer = factory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.ReLU);
            Assert.Equal(ActivationFunctions.ReLU, reluLayer.Neurons[0].ActivationFunction);
        }

        [Fact]
        public void LayerFactory_ImplementsILayerFactory()
        {
            // Arrange & Act
            ILayerFactory factory = new LayerFactory();

            // Assert
            Assert.IsAssignableFrom<ILayerFactory>(factory);
        }

        #endregion

        #region Edge Cases and Performance Tests

        [Fact]
        public void Layer_Forward_WithZeroInputs_ReturnsOnlyBiases()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] biases = { new double[] { 0.5 }, new double[] { -0.3 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 0.0, 0.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(2, outputs.Length);
            Assert.Equal(0.5, outputs[0], Tolerance);
            Assert.Equal(-0.3, outputs[1], Tolerance);
        }

        [Fact]
        public void Layer_Forward_WithNegativeInputs_ProcessesCorrectly()
        {
            // Arrange
            double[][] weights = { new double[] { 1.0, -1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { -2.0, 3.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(-5.0, outputs[0], Tolerance); // (-2.0)*1.0 + 3.0*(-1.0) = -5.0
        }

        [Fact]
        public void Layer_Forward_ConsistentResults_MultipleCalls()
        {
            // Arrange
            double[][] weights = { new double[] { 0.5, 0.3 } };
            double[][] biases = { new double[] { 0.1 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);
            double[] inputs = { 1.0, 2.0 };

            // Act
            double[] outputs1 = layer.Forward(inputs);
            double[] outputs2 = layer.Forward(inputs);
            double[] outputs3 = layer.Forward(inputs);

            // Assert
            Assert.Equal(outputs1[0], outputs2[0], Tolerance);
            Assert.Equal(outputs2[0], outputs3[0], Tolerance);
        }

        [Fact]
        public void Layer_Forward_WithLargeNumberOfNeurons_ProcessesEfficiently()
        {
            // Arrange
            int neuronCount = 100;
            double[][] weights = Enumerable.Range(0, neuronCount)
                .Select(i => new double[] { (double)i / neuronCount })
                .ToArray();
            double[][] biases = Enumerable.Range(0, neuronCount)
                .Select(i => new double[] { (double)i / neuronCount })
                .ToArray();
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { 1.0 };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Equal(neuronCount, outputs.Length);
            for (int i = 0; i < neuronCount; i++)
            {
                double expected = 1.0 * (i / (double)neuronCount) + (i / (double)neuronCount);
                Assert.Equal(expected, outputs[i], Tolerance);
            }
        }

        [Theory]
        [InlineData(double.MaxValue)]
        [InlineData(double.MinValue)]
        [InlineData(double.Epsilon)]
        [InlineData(-double.Epsilon)]
        public void Layer_Forward_WithExtremeValues_HandlesCorrectly(double extremeValue)
        {
            // Arrange
            double[][] weights = { new double[] { 1.0 } };
            double[][] biases = { new double[] { 0.0 } };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            double[] inputs = { extremeValue };

            // Act
            double[] outputs = layer.Forward(inputs);

            // Assert
            Assert.Single(outputs);
            Assert.Equal(extremeValue, outputs[0]);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void Layer_IntegrationTest_MultipleForwardPasses()
        {
            // Arrange
            double[][] weights = { 
                new double[] { 0.5, 0.2 }, 
                new double[] { -0.3, 0.8 } 
            };
            double[][] biases = { 
                new double[] { 0.1 }, 
                new double[] { -0.2 } 
            };
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer = new Layer(0, nodeFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Sigmoid);

            // Act & Assert - Multiple different inputs
            double[] inputs1 = { 1.0, 2.0 };
            double[] outputs1 = layer.Forward(inputs1);
            Assert.Equal(2, outputs1.Length);
            Assert.True(outputs1[0] >= 0 && outputs1[0] <= 1); // Sigmoid range
            Assert.True(outputs1[1] >= 0 && outputs1[1] <= 1); // Sigmoid range

            double[] inputs2 = { -1.0, 0.5 };
            double[] outputs2 = layer.Forward(inputs2);
            Assert.Equal(2, outputs2.Length);
            Assert.True(outputs2[0] >= 0 && outputs2[0] <= 1); // Sigmoid range
            Assert.True(outputs2[1] >= 0 && outputs2[1] <= 1); // Sigmoid range

            // Outputs should be different for different inputs
            Assert.NotEqual(outputs1[0], outputs2[0]);
            Assert.NotEqual(outputs1[1], outputs2[1]);
        }

        [Fact]
        public void Layer_ChainedLayers_DataFlowsCorrectly()
        {
            // Arrange
            double[][] weights1 = { new double[] { 1.0 }, new double[] { -1.0 } };
            double[][] biases1 = { new double[] { 0.0 }, new double[] { 0.0 } };
            double[][] weights2 = { new double[] { 0.5, 0.5 } };
            double[][] biases2 = { new double[] { 0.0 } };
            
            var nodeFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            var layer1 = new Layer(0, nodeFactory, inputProcessorFactory, weights1, biases1, ActivationFunctions.Unit);
            var layer2 = new Layer(1, nodeFactory, inputProcessorFactory, weights2, biases2, ActivationFunctions.Unit);
            
            // Set up layer connections
            layer1.NextLayer = layer2;
            layer2.PreviousLayer = layer1;

            // Act
            double[] inputs = { 2.0 };
            double[] layer1Outputs = layer1.Forward(inputs);
            double[] layer2Outputs = layer2.Forward(layer1Outputs);

            // Assert
            Assert.Equal(2, layer1Outputs.Length);
            Assert.Equal(2.0, layer1Outputs[0], Tolerance);  // 2.0*1.0 = 2.0
            Assert.Equal(-2.0, layer1Outputs[1], Tolerance); // 2.0*(-1.0) = -2.0
            
            Assert.Single(layer2Outputs);
            Assert.Equal(0.0, layer2Outputs[0], Tolerance); // 2.0*0.5 + (-2.0)*0.5 = 0.0
        }

        #endregion
    }
}
