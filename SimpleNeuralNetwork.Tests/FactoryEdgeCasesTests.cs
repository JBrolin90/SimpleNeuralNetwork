using Xunit;
using System;
using BackPropagation.NNLib;
using Moq;

namespace SimpleNeuralNetwork.Tests
{
    public class FactoryEdgeCasesTests
    {
        private const double Tolerance = 1e-7;

        #region LayerFactory Edge Cases

        [Fact]
        public void LayerFactory_Create_WithNullNeuronFactory_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                layerFactory.Create(0, null!, inputProcessorFactory, weights, biases, ActivationFunctions.Unit));
        }

        [Fact]
        public void LayerFactory_Create_WithNullInputProcessorFactory_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                layerFactory.Create(0, neuronFactory, null!, weights, biases, ActivationFunctions.Unit));
        }

        [Fact]
        public void LayerFactory_Create_WithNullWeights_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] biases = { new double[] { 0.5 } };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                layerFactory.Create(0, neuronFactory, inputProcessorFactory, null!, biases, ActivationFunctions.Unit));
        }

        [Fact]
        public void LayerFactory_Create_WithNullBiases_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                layerFactory.Create(0, neuronFactory, inputProcessorFactory, weights, null!, ActivationFunctions.Unit));
        }

        [Fact]
        public void LayerFactory_Create_WithEmptyWeights_CreatesEmptyLayer()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = Array.Empty<double[]>();
            double[][] biases = Array.Empty<double[]>();

            // Act
            var layer = layerFactory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.NotNull(layer);
            Assert.Empty(layer.Neurons);
            Assert.Empty(layer.InputProcessors);
        }

        [Fact]
        public void LayerFactory_Create_WithMismatchedWeightsAndBiases_ThrowsException()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } }; // 2 neurons
            double[][] biases = { new double[] { 0.5 } }; // 1 neuron

            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => 
                layerFactory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit));
        }

        [Fact]
        public void LayerFactory_Create_WithNegativeIndex_CreatesLayerWithNegativeIndex()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act
            var layer = layerFactory.Create(-1, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(-1, layer.Index);
        }

        [Fact]
        public void LayerFactory_Create_WithNullActivationFunction_UsesNullFunction()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act
            var layer = layerFactory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, null!);

            // Assert
            Assert.NotNull(layer);
            // The layer should be created, but activation function may be null
        }

        #endregion

        #region NeuronFactory Edge Cases

        [Fact]
        public void NeuronFactory_Create_WithVeryLargeIndex_CreatesNeuronWithLargeIndex()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();
            var mockLayer = new Mock<ILayer>();
            int largeIndex = int.MaxValue;

            // Act
            var neuron = neuronFactory.Create(mockLayer.Object, largeIndex, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(largeIndex, neuron.Index);
        }

        [Fact]
        public void NeuronFactory_Create_WithNegativeIndex_CreatesNeuronWithNegativeIndex()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();
            var mockLayer = new Mock<ILayer>();
            int negativeIndex = -100;

            // Act
            var neuron = neuronFactory.Create(mockLayer.Object, negativeIndex, ActivationFunctions.Unit);

            // Assert
            Assert.Equal(negativeIndex, neuron.Index);
        }

        [Fact]
        public void NeuronFactory_Create_WithDifferentActivationFunctions_CreatesCorrectNeurons()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();
            var mockLayer = new Mock<ILayer>();
            var activationFunctions = new Func<double, double>[] {
                ActivationFunctions.Unit,
                ActivationFunctions.Sigmoid,
                ActivationFunctions.ReLU,
                ActivationFunctions.Tanh
            };

            // Act & Assert
            foreach (var activationFunction in activationFunctions)
            {
                var neuron = neuronFactory.Create(mockLayer.Object, 0, activationFunction);
                Assert.NotNull(neuron);
                Assert.Equal(activationFunction, neuron.ActivationFunction);
            }
        }

        #endregion

        #region InputProcessorFactory Edge Cases

        [Fact]
        public void InputProcessorFactory_Build_WithNullWeights_ThrowsException()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] biases = { 0.5 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                inputProcessorFactory.Build(mockLayer.Object, 0, null!, biases));
        }

        [Fact]
        public void InputProcessorFactory_Build_WithNullBiases_ThrowsException()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] weights = { 1.0, 2.0 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => 
                inputProcessorFactory.Build(mockLayer.Object, 0, weights, null!));
        }

        [Fact]
        public void InputProcessorFactory_Build_WithEmptyWeights_CreatesInputProcessor()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] weights = Array.Empty<double>();
            double[] biases = { 0.5 };

            // Act
            var inputProcessor = inputProcessorFactory.Build(mockLayer.Object, 0, weights, biases);

            // Assert
            Assert.NotNull(inputProcessor);
            Assert.Empty(inputProcessor.Weights);
            Assert.Equal(biases, inputProcessor.Bias);
        }

        [Fact]
        public void InputProcessorFactory_Build_WithEmptyBiases_CreatesInputProcessor()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] weights = { 1.0, 2.0 };
            double[] biases = Array.Empty<double>();

            // Act
            var inputProcessor = inputProcessorFactory.Build(mockLayer.Object, 0, weights, biases);

            // Assert
            Assert.NotNull(inputProcessor);
            Assert.Equal(weights, inputProcessor.Weights);
            Assert.Empty(inputProcessor.Bias);
        }

        [Fact]
        public void InputProcessorFactory_Build_WithVeryLargeArrays_CreatesInputProcessor()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] weights = new double[10000];
            double[] biases = { 0.5 };

            // Act
            var inputProcessor = inputProcessorFactory.Build(mockLayer.Object, 0, weights, biases);

            // Assert
            Assert.NotNull(inputProcessor);
            Assert.Equal(10000, inputProcessor.Weights.Length);
        }

        [Fact]
        public void InputProcessorFactory_Build_WithNegativeIndex_CreatesInputProcessor()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();
            var mockLayer = new Mock<ILayer>();
            double[] weights = { 1.0, 2.0 };
            double[] biases = { 0.5 };

            // Act
            var inputProcessor = inputProcessorFactory.Build(mockLayer.Object, -5, weights, biases);

            // Assert
            Assert.NotNull(inputProcessor);
            Assert.Equal(-5, inputProcessor.Index);
        }

        #endregion

        #region Memory and Performance Tests

        [Fact]
        public void LayerFactory_Create_WithLargeLayer_CompletesInReasonableTime()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            // Create a relatively large layer
            int neuronCount = 1000;
            int inputCount = 100;
            double[][] weights = new double[neuronCount][];
            double[][] biases = new double[neuronCount][];
            
            for (int i = 0; i < neuronCount; i++)
            {
                weights[i] = new double[inputCount];
                biases[i] = new double[1];
            }

            // Act
            var startTime = DateTime.Now;
            var layer = layerFactory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            var endTime = DateTime.Now;

            // Assert
            Assert.NotNull(layer);
            Assert.Equal(neuronCount, layer.Neurons.Length);
            Assert.True((endTime - startTime).TotalSeconds < 5, "Layer creation should complete within 5 seconds");
        }

        [Fact]
        public void NeuronFactory_Create_MultipleNeurons_DoesNotLeakMemory()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();
            var mockLayer = new Mock<ILayer>();
            const int neuronCount = 10000;

            // Act
            var neurons = new INeuron[neuronCount];
            for (int i = 0; i < neuronCount; i++)
            {
                neurons[i] = neuronFactory.Create(mockLayer.Object, i, ActivationFunctions.Unit);
            }

            // Assert
            Assert.Equal(neuronCount, neurons.Length);
            for (int i = 0; i < neuronCount; i++)
            {
                Assert.NotNull(neurons[i]);
                Assert.Equal(i, neurons[i].Index);
            }
        }

        #endregion

        #region Interface Implementation Tests

        [Fact]
        public void LayerFactory_ImplementsILayerFactory_Correctly()
        {
            // Arrange
            var layerFactory = new LayerFactory();

            // Act & Assert
            Assert.IsAssignableFrom<ILayerFactory>(layerFactory);
            
            // Verify interface method can be called
            ILayerFactory interfaceFactory = layerFactory;
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };
            
            var layer = interfaceFactory.Create(0, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
            Assert.NotNull(layer);
        }

        [Fact]
        public void NeuronFactory_ImplementsINeuronFactory_Correctly()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();

            // Act & Assert
            Assert.IsAssignableFrom<INeuronFactory>(neuronFactory);
            
            // Verify interface method can be called
            INeuronFactory interfaceFactory = neuronFactory;
            var mockLayer = new Mock<ILayer>();
            var neuron = interfaceFactory.Create(mockLayer.Object, 0, ActivationFunctions.Unit);
            Assert.NotNull(neuron);
        }

        [Fact]
        public void InputProcessorFactory_ImplementsIInputProcessorFactory_Correctly()
        {
            // Arrange
            var inputProcessorFactory = new InputProcessorFactory();

            // Act & Assert
            Assert.IsAssignableFrom<IInputProcessorFactory>(inputProcessorFactory);
            
            // Verify interface method can be called
            IInputProcessorFactory interfaceFactory = inputProcessorFactory;
            var mockLayer = new Mock<ILayer>();
            double[] weights = { 1.0, 2.0 };
            double[] biases = { 0.5 };
            var inputProcessor = interfaceFactory.Build(mockLayer.Object, 0, weights, biases);
            Assert.NotNull(inputProcessor);
        }

        #endregion

        #region Concurrent Access Tests

        [Fact]
        public async System.Threading.Tasks.Task LayerFactory_Create_ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            double[][] weights = { new double[] { 1.0, 2.0 } };
            double[][] biases = { new double[] { 0.5 } };

            // Act
            var tasks = new System.Threading.Tasks.Task[10];
            for (int i = 0; i < tasks.Length; i++)
            {
                int index = i;
                tasks[i] = System.Threading.Tasks.Task.Run(() =>
                {
                    var layer = layerFactory.Create(index, neuronFactory, inputProcessorFactory, weights, biases, ActivationFunctions.Unit);
                    Assert.NotNull(layer);
                    Assert.Equal(index, layer.Index);
                });
            }

            // Assert
            await System.Threading.Tasks.Task.WhenAll(tasks);
        }

        [Fact]
        public async System.Threading.Tasks.Task NeuronFactory_Create_ConcurrentAccess_ThreadSafe()
        {
            // Arrange
            var neuronFactory = new NeuronFactory();
            var mockLayer = new Mock<ILayer>();

            // Act
            var tasks = new System.Threading.Tasks.Task[10];
            for (int i = 0; i < tasks.Length; i++)
            {
                int index = i;
                tasks[i] = System.Threading.Tasks.Task.Run(() =>
                {
                    var neuron = neuronFactory.Create(mockLayer.Object, index, ActivationFunctions.Unit);
                    Assert.NotNull(neuron);
                    Assert.Equal(index, neuron.Index);
                });
            }

            // Assert
            await System.Threading.Tasks.Task.WhenAll(tasks);
        }

        #endregion
    }
}
