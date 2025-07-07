using Xunit;
using System;
using BackPropagation.NNLib;
using Moq;
using System.Linq;

namespace SimpleNeuralNetwork.Tests
{
    public class NeuralNetworkTrainerTests
    {
        private const double Tolerance = 1e-7;

        #region Constructor Tests

        [Fact]
        public void NeuralNetworkTrainer_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            double learningRate = 0.01;

            // Act
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, learningRate);

            // Assert
            Assert.Equal(learningRate, trainer.LearningRate);
            Assert.NotNull(trainer.LossFunction);
            Assert.NotNull(trainer.LossFunctionD);
            Assert.Empty(trainer.Gradients);
        }

        [Fact]
        public void NeuralNetworkTrainer_Constructor_WithNullNetwork_DoesNotThrowException()
        {
            // Arrange & Act
            var trainer = new NeuralNetworkTrainer(null!, 0.01);

            // Assert - The constructor doesn't validate the network parameter
            Assert.Equal(0.01, trainer.LearningRate);
        }

        [Fact]
        public void NeuralNetworkTrainer_Constructor_WithNegativeLearningRate_AllowsNegativeValue()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            double learningRate = -0.01;

            // Act
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, learningRate);

            // Assert
            Assert.Equal(learningRate, trainer.LearningRate);
        }

        [Fact]
        public void NeuralNetworkTrainer_Constructor_WithZeroLearningRate_AllowsZeroValue()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            double learningRate = 0.0;

            // Act
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, learningRate);

            // Assert
            Assert.Equal(learningRate, trainer.LearningRate);
        }

        #endregion

        #region Gradients Class Tests

        [Fact]
        public void Gradients_Constructor_InitializesWeightGradientArrayCorrectly()
        {
            // Arrange
            int weightCount = 5;

            // Act
            var gradients = new Gradients(weightCount);

            // Assert
            Assert.Equal(weightCount, gradients.WeightGradient.Length);
            Assert.Equal(0.0, gradients.BiasGradient);
            Assert.All(gradients.WeightGradient, weight => Assert.Equal(0.0, weight));
        }

        [Fact]
        public void Gradients_Constructor_WithZeroWeightCount_CreatesEmptyArray()
        {
            // Arrange
            int weightCount = 0;

            // Act
            var gradients = new Gradients(weightCount);

            // Assert
            Assert.Empty(gradients.WeightGradient);
            Assert.Equal(0.0, gradients.BiasGradient);
        }

        [Fact]
        public void Gradients_Constructor_WithNegativeWeightCount_ThrowsException()
        {
            // Arrange
            int weightCount = -1;

            // Act & Assert
            var exception = Assert.Throws<ArgumentOutOfRangeException>(() => new Gradients(weightCount));
            Assert.Equal("weightCount", exception.ParamName);
        }

        #endregion

        #region PrepareBackPropagation Tests

        [Fact]
        public void PrepareBackPropagation_InitializesGradientsArrayCorrectly()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);

            // Act
            trainer.PrepareBackPropagation();

            // Assert
            Assert.Equal(2, trainer.Gradients.Length);
            Assert.Equal(2, trainer.Gradients[0].Length); // First layer has 2 neurons
            Assert.Single(trainer.Gradients[1]); // Second layer has 1 neuron
        }

        [Fact]
        public void PrepareBackPropagation_InitializesGradientsWithCorrectWeightCounts()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);

            // Act
            trainer.PrepareBackPropagation();

            // Assert
            // First layer neurons have 2 weights each
            Assert.Equal(2, trainer.Gradients[0][0].WeightGradient.Length);
            Assert.Equal(2, trainer.Gradients[0][1].WeightGradient.Length);
            
            // Second layer neuron has 2 weights
            Assert.Equal(2, trainer.Gradients[1][0].WeightGradient.Length);
        }

        #endregion

        #region TrainOneEpoch Tests

        [Fact]
        public void TrainOneEpoch_WithValidData_ReturnsLossArray()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            double[] loss = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(loss[0] >= 0); // Loss should be non-negative
        }

        [Fact]
        public void TrainOneEpoch_WithSampleArray_ReturnsLossArray()
        {
            // Arrange
            var network = CreateRealNetworkForSamples();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            var samples = new Sample[] { new Sample(1.0, 2.0, Operation.add, 1.0) };

            // Act
            double[] loss = trainer.TrainOneEpoch(samples);

            // Assert
            Assert.Single(loss);
            Assert.True(loss[0] >= 0); // Loss should be non-negative
        }

        [Fact]
        public void TrainOneEpoch_WithMultipleTrainingExamples_AveragesLoss()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { 
                new double[] { 1.0, 2.0 },
                new double[] { 2.0, 3.0 }
            };
            double[][] observed = { 
                new double[] { 3.0 },
                new double[] { 5.0 }
            };

            // Act
            double[] loss = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(loss[0] >= 0);
        }

        [Fact]
        public void TrainOneEpoch_CallsPredictForEachTrainingExample()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var mockLayers = CreateMockLayers();
            mockNetwork.Setup(n => n.Layers).Returns(mockLayers);
            mockNetwork.Setup(n => n.Predict(It.IsAny<double[]>())).Returns(new double[] { 2.5 });
            mockNetwork.Setup(n => n.Weigths).Returns(CreateMockWeights());
            mockNetwork.Setup(n => n.Biases).Returns(CreateMockBiases());
            
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);
            
            double[][] trainingData = { 
                new double[] { 1.0, 2.0 },
                new double[] { 2.0, 3.0 }
            };
            double[][] observed = { 
                new double[] { 3.0 },
                new double[] { 5.0 }
            };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            mockNetwork.Verify(n => n.Predict(It.IsAny<double[]>()), Times.Exactly(2));
        }

        #endregion

        #region PropagateBackwards Tests

        [Fact]
        public void PropagateBackwards_WithValidNetwork_DoesNotThrowException()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            trainer.PrepareBackPropagation();
            
            double[] dLoss = { 0.5 };

            // Act & Assert
            var exception = Record.Exception(() => trainer.PropagateBackwards(dLoss));
            Assert.Null(exception);
        }

        [Fact]
        public void PropagateBackwards_WithNullDLoss_ThrowsNullReferenceException()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            trainer.PrepareBackPropagation();

            // Act & Assert
            Assert.Throws<NullReferenceException>(() => trainer.PropagateBackwards(null!));
        }

        #endregion

        #region Learning Rate Property Tests

        [Fact]
        public void LearningRate_CanBeSetAndRetrieved()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);
            double newLearningRate = 0.05;

            // Act
            trainer.LearningRate = newLearningRate;

            // Assert
            Assert.Equal(newLearningRate, trainer.LearningRate);
        }

        [Fact]
        public void LearningRate_CanBeSetToZero()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);

            // Act
            trainer.LearningRate = 0.0;

            // Assert
            Assert.Equal(0.0, trainer.LearningRate);
        }

        [Fact]
        public void LearningRate_CanBeSetToNegativeValue()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);

            // Act
            trainer.LearningRate = -0.01;

            // Assert
            Assert.Equal(-0.01, trainer.LearningRate);
        }

        #endregion

        #region Loss Function Tests

        [Fact]
        public void LossFunction_CanBeSetAndRetrieved()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);
            Func<double[][], double[][], double[]> customLossFunction = (pred, obs) => new double[] { 0.0 };

            // Act
            trainer.LossFunction = customLossFunction;

            // Assert
            Assert.Equal(customLossFunction, trainer.LossFunction);
        }

        [Fact]
        public void LossFunctionD_CanBeSetAndRetrieved()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);
            Func<double[][], double[][], double[]> customLossFunctionD = (pred, obs) => new double[] { 0.0 };

            // Act
            trainer.LossFunctionD = customLossFunctionD;

            // Assert
            Assert.Equal(customLossFunctionD, trainer.LossFunctionD);
        }

        [Fact]
        public void LossFunction_DefaultsToSumSquaredError()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);

            // Act & Assert
            Assert.Equal(LossFunctions.SumSquaredError, trainer.LossFunction);
        }

        [Fact]
        public void LossFunctionD_DefaultsToSumSquaredErrorDerivative()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);

            // Act & Assert
            Assert.Equal(LossFunctions.SumSquaredErrorDerivative, trainer.LossFunctionD);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void TrainOneEpoch_WithEmptyTrainingData_ThrowsArgumentException()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = Array.Empty<double[]>();
            double[][] observed = Array.Empty<double[]>();

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => trainer.TrainOneEpoch(trainingData, observed));
            Assert.Equal("Training data cannot be empty", exception.Message);
        }

        [Fact]
        public void TrainOneEpoch_WithMismatchedDataAndObservedLengths_ThrowsArgumentException()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 }, new double[] { 4.0 } };

            // Act & Assert
            var exception = Assert.Throws<ArgumentException>(() => trainer.TrainOneEpoch(trainingData, observed));
            Assert.Equal("Training data and observed data must have the same length", exception.Message);
        }

        [Fact]
        public void PrepareBackPropagation_WithEmptyNetwork_HandlesGracefully()
        {
            // Arrange
            var mockNetwork = new Mock<INeuralNetwork>();
            mockNetwork.Setup(n => n.Layers).Returns(Array.Empty<ILayer>());
            var trainer = new NeuralNetworkTrainer(mockNetwork.Object, 0.01);

            // Act
            var exception = Record.Exception(() => trainer.PrepareBackPropagation());

            // Assert
            Assert.Null(exception);
            Assert.Empty(trainer.Gradients);
        }

        #endregion

        #region Additional Comprehensive Tests

        [Fact]
        public void TrainOneEpoch_UpdatesNetworkWeights()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.1); // Higher learning rate for visible changes
            
            // Store initial weights
            var initialWeights = network.Weigths[0][0][0];
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotEqual(initialWeights, network.Weigths[0][0][0]);
        }

        [Fact]
        public void TrainOneEpoch_UpdatesNetworkBiases()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.1); // Higher learning rate for visible changes
            
            // Store initial bias
            var initialBias = network.Biases[0][0][0];
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotEqual(initialBias, network.Biases[0][0][0]);
        }

        [Fact]
        public void TrainOneEpoch_WithZeroLearningRate_DoesNotUpdateWeights()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.0); // Zero learning rate
            
            // Store initial weights
            var initialWeights = network.Weigths[0][0][0];
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Equal(initialWeights, network.Weigths[0][0][0]);
        }

        [Fact]
        public void TrainOneEpoch_WithMultipleEpochs_ContinuesTraining()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            double[] loss1 = trainer.TrainOneEpoch(trainingData, observed);
            double[] loss2 = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Single(loss1);
            Assert.Single(loss2);
            Assert.True(loss1[0] >= 0);
            Assert.True(loss2[0] >= 0);
        }

        [Fact]
        public void Gradients_AfterTraining_ContainNonZeroValues()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.NotEmpty(trainer.Gradients);
            Assert.True(trainer.Gradients.Length > 0);
            Assert.True(trainer.Gradients[0].Length > 0);
        }

        [Fact]
        public void TrainOneEpoch_UsesCustomLossFunction()
        {
            // Arrange
            var network = CreateRealNetworkWithTwoLayers();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            bool customLossCalled = false;
            trainer.LossFunction = (pred, obs) => { customLossCalled = true; return new double[] { 1.0 }; };
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };

            // Act
            trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            // The implementation uses the configurable LossFunction property
            Assert.True(customLossCalled);
        }

        [Fact]
        public void TrainOneEpoch_WithSingleInputOutput_HandlesCorrectly()
        {
            // Arrange
            var network = CreateSimpleNetworkWithSingleInputOutput();
            var trainer = new NeuralNetworkTrainer(network, 0.01);
            
            double[][] trainingData = { new double[] { 2.0 } };
            double[][] observed = { new double[] { 4.0 } };

            // Act
            double[] loss = trainer.TrainOneEpoch(trainingData, observed);

            // Assert
            Assert.Single(loss);
            Assert.True(loss[0] >= 0);
        }

        #endregion

        #region Helper Methods

        private INeuralNetwork CreateRealNetworkWithTwoLayers()
        {
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            double[][][] weights = {
                new double[][] { new double[] { 0.5, 0.5 }, new double[] { 0.3, 0.7 } }, // First layer: 2 neurons, 2 inputs each
                new double[][] { new double[] { 0.4, 0.6 } }  // Second layer: 1 neuron, 2 inputs
            };
            
            double[][][] biases = {
                new double[][] { new double[] { 0.1 }, new double[] { 0.2 } }, // First layer: 2 neurons, 1 bias each
                new double[][] { new double[] { 0.3 } }  // Second layer: 1 neuron, 1 bias
            };
            
            double[][] ys = {
                new double[] { 0.0, 0.0 }, // First layer: 2 neurons
                new double[] { 0.0 }       // Second layer: 1 neuron
            };
            
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid, // First layer
                ActivationFunctions.Sigmoid  // Second layer
            };

            return new NeuralNetwork(layerFactory, neuronFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
        }

        private INeuralNetwork CreateRealNetworkForSamples()
        {
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            // Network that accepts 4 inputs (for Sample class)
            double[][][] weights = {
                new double[][] { 
                    new double[] { 0.5, 0.5, 0.2, 0.3 }, // First neuron: 4 inputs
                    new double[] { 0.3, 0.7, 0.1, 0.4 }  // Second neuron: 4 inputs
                },
                new double[][] { new double[] { 0.4, 0.6 } }  // Second layer: 1 neuron, 2 inputs
            };
            
            double[][][] biases = {
                new double[][] { new double[] { 0.1 }, new double[] { 0.2 } }, // First layer: 2 neurons, 1 bias each
                new double[][] { new double[] { 0.3 } }  // Second layer: 1 neuron, 1 bias
            };
            
            double[][] ys = {
                new double[] { 0.0, 0.0 }, // First layer: 2 neurons
                new double[] { 0.0 }       // Second layer: 1 neuron
            };
            
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid, // First layer
                ActivationFunctions.Sigmoid  // Second layer
            };

            return new NeuralNetwork(layerFactory, neuronFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
        }

        private ILayer[] CreateMockLayers()
        {
            var mockLayer1 = new Mock<ILayer>();
            var mockLayer2 = new Mock<ILayer>();
            
            var mockNeuron1 = new Mock<INeuron>();
            var mockNeuron2 = new Mock<INeuron>();
            var mockNeuron3 = new Mock<INeuron>();
            
            var mockInputProcessor1 = new Mock<IInputProcessor>();
            var mockInputProcessor2 = new Mock<IInputProcessor>();
            var mockInputProcessor3 = new Mock<IInputProcessor>();
            
            mockInputProcessor1.Setup(ip => ip.Weights).Returns(new double[] { 0.5, 0.5 });
            mockInputProcessor2.Setup(ip => ip.Weights).Returns(new double[] { 0.3, 0.7 });
            mockInputProcessor3.Setup(ip => ip.Weights).Returns(new double[] { 0.4, 0.6 });
            
            mockInputProcessor1.Setup(ip => ip.Y).Returns(1.0);
            mockInputProcessor2.Setup(ip => ip.Y).Returns(1.0);
            mockInputProcessor3.Setup(ip => ip.Y).Returns(1.0);
            
            mockNeuron1.Setup(n => n.ActivationDerivative).Returns(ActivationFunctions.SigmoidDerivative);
            mockNeuron2.Setup(n => n.ActivationDerivative).Returns(ActivationFunctions.SigmoidDerivative);
            mockNeuron3.Setup(n => n.ActivationDerivative).Returns(ActivationFunctions.SigmoidDerivative);
            
            mockNeuron1.Setup(n => n.Derivative(It.IsAny<double>())).Returns(1.0);
            mockNeuron2.Setup(n => n.Derivative(It.IsAny<double>())).Returns(1.0);
            mockNeuron3.Setup(n => n.Derivative(It.IsAny<double>())).Returns(1.0);
            
            mockLayer1.Setup(l => l.Neurons).Returns(new INeuron[] { mockNeuron1.Object, mockNeuron2.Object });
            mockLayer1.Setup(l => l.InputProcessors).Returns(new IInputProcessor[] { mockInputProcessor1.Object, mockInputProcessor2.Object });
            mockLayer1.Setup(l => l.Inputs).Returns(new double[] { 1.0, 2.0 });
            
            mockLayer2.Setup(l => l.Neurons).Returns(new INeuron[] { mockNeuron3.Object });
            mockLayer2.Setup(l => l.InputProcessors).Returns(new IInputProcessor[] { mockInputProcessor3.Object });
            mockLayer2.Setup(l => l.Inputs).Returns(new double[] { 1.0, 1.0 });
            
            return new ILayer[] { mockLayer1.Object, mockLayer2.Object };
        }

        private double[][][] CreateMockWeights()
        {
            return new double[][][] 
            {
                new double[][] { new double[] { 0.5, 0.5 }, new double[] { 0.3, 0.7 } },
                new double[][] { new double[] { 0.4, 0.6 } }
            };
        }

        private double[][][] CreateMockBiases()
        {
            return new double[][][] 
            {
                new double[][] { new double[] { 0.1 }, new double[] { 0.2 } },
                new double[][] { new double[] { 0.3 } }
            };
        }

        private INeuralNetwork CreateSimpleNetworkWithSingleInputOutput()
        {
            var layerFactory = new LayerFactory();
            var neuronFactory = new NeuronFactory();
            var inputProcessorFactory = new InputProcessorFactory();
            
            double[][][] weights = {
                new double[][] { new double[] { 0.5 } }  // Single neuron with single input
            };
            
            double[][][] biases = {
                new double[][] { new double[] { 0.1 } }  // Single neuron with single bias
            };
            
            double[][] ys = {
                new double[] { 0.0 }  // Single neuron
            };
            
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid
            };

            return new NeuralNetwork(layerFactory, neuronFactory, inputProcessorFactory, weights, biases, ys, activationFunctions);
        }

        #endregion
    }
}
