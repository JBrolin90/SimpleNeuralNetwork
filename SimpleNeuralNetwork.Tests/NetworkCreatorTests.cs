using Xunit;
using System;
using BackPropagation.NNLib;
using System.Linq;

namespace SimpleNeuralNetwork.Tests
{
    public class NetworkCreatorTests
    {
        private const double Tolerance = 1e-7;

        #region Constructor Tests

        [Fact]
        public void NetworkCreator_Constructor_InitializesWeightsArrayCorrectly()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 3, 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(2, creator.Weights.Length); // 2 layers
            Assert.Equal(3, creator.Weights[0].Length); // First layer has 3 neurons
            Assert.Equal(2, creator.Weights[1].Length); // Second layer has 2 neurons
            Assert.Equal(2, creator.Weights[0][0].Length); // First layer neurons have 2 inputs
            Assert.Equal(3, creator.Weights[1][0].Length); // Second layer neurons have 3 inputs (from previous layer)
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesBiasesArrayCorrectly()
        {
            // Arrange
            int inputs = 3;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit, ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(2, creator.Biases.Length); // 2 layers
            Assert.Equal(2, creator.Biases[0].Length); // First layer has 2 neurons
            Assert.Single(creator.Biases[1]); // Second layer has 1 neuron
            Assert.Single(creator.Biases[0][0]); // Each neuron has 1 bias
            Assert.Single(creator.Biases[1][0]); // Each neuron has 1 bias
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesYsArrayCorrectly()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 4, 3, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(3, creator.Ys.Length); // 3 layers
            Assert.Equal(4, creator.Ys[0].Length); // First layer has 4 neurons
            Assert.Equal(3, creator.Ys[1].Length); // Second layer has 3 neurons
            Assert.Single(creator.Ys[2]); // Third layer has 1 neuron
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesWeightsToZero()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit, ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            foreach (var layer in creator.Weights)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        Assert.Equal(0.0, weight);
                    }
                }
            }
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesBiasesToZero()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit, ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            foreach (var layer in creator.Biases)
            {
                foreach (var neuron in layer)
                {
                    foreach (var bias in neuron)
                    {
                        Assert.Equal(0.0, bias);
                    }
                }
            }
        }

        [Fact]
        public void NetworkCreator_Constructor_StoresActivationFunctions()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(activationFunctions, creator.ActivationFunctions);
        }

        [Fact]
        public void NetworkCreator_Constructor_WithSingleLayer_InitializesCorrectly()
        {
            // Arrange
            int inputs = 3;
            int[] layerSizes = { 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Single(creator.Weights);
            Assert.Single(creator.Biases);
            Assert.Single(creator.Ys);
            Assert.Single(creator.Weights[0]);
            Assert.Equal(3, creator.Weights[0][0].Length);
        }

        [Fact]
        public void NetworkCreator_Constructor_WithEmptyLayerSizes_CreatesEmptyArrays()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = Array.Empty<int>();
            Func<double, double>[] activationFunctions = Array.Empty<Func<double, double>>();

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Empty(creator.Weights);
            Assert.Empty(creator.Biases);
            Assert.Empty(creator.Ys);
        }

        [Fact]
        public void NetworkCreator_Constructor_WithZeroInputs_HandlesCorrectly()
        {
            // Arrange
            int inputs = 0;
            int[] layerSizes = { 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Single(creator.Weights);
            Assert.Single(creator.Weights[0]);
            Assert.Empty(creator.Weights[0][0]);
        }

        #endregion

        #region RandomizeWeights Tests

        [Fact]
        public void RandomizeWeights_ModifiesWeightsWithinRange()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 3, 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            double from = -1.0;
            double to = 1.0;

            // Act
            creator.RandomizeWeights(from, to);

            // Assert
            foreach (var layer in creator.Weights)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        Assert.InRange(weight, from, to);
                    }
                }
            }
        }

        [Fact]
        public void RandomizeWeights_WithNegativeRange_ModifiesWeightsCorrectly()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            double from = -2.0;
            double to = -1.0;

            // Act
            creator.RandomizeWeights(from, to);

            // Assert
            foreach (var layer in creator.Weights)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        Assert.InRange(weight, from, to);
                    }
                }
            }
        }

        [Fact]
        public void RandomizeWeights_WithZeroRange_SetsWeightsToZero()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            double from = 0.0;
            double to = 0.0;

            // Act
            creator.RandomizeWeights(from, to);

            // Assert
            foreach (var layer in creator.Weights)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        Assert.Equal(0.0, weight);
                    }
                }
            }
        }

        [Fact]
        public void RandomizeWeights_ChangesWeightsFromInitialValues()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            
            // Store initial weights (should be 0)
            double initialWeight = creator.Weights[0][0][0];
            
            // Act
            creator.RandomizeWeights(-1.0, 1.0);

            // Assert
            // At least one weight should be different from initial value
            bool hasChangedWeight = false;
            foreach (var layer in creator.Weights)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        if (Math.Abs(weight - initialWeight) > Tolerance)
                        {
                            hasChangedWeight = true;
                            break;
                        }
                    }
                }
            }
            Assert.True(hasChangedWeight, "RandomizeWeights should change at least one weight from initial value");
        }

        #endregion

        #region Static Methods Tests

        [Fact]
        public void ActOn3dArr_ExecutesActionOnAllElements()
        {
            // Arrange
            double[][][] array = {
                new double[][] { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } },
                new double[][] { new double[] { 5.0, 6.0 } }
            };
            int actionCount = 0;
            Action<double> countAction = (x) => actionCount++;

            // Act
            NetworkCreator.ActOn3dArr(array, countAction);

            // Assert
            Assert.Equal(6, actionCount); // Should execute on all 6 elements
        }

        [Fact]
        public void ActOn3dArr_WithEmptyArray_DoesNotThrow()
        {
            // Arrange
            double[][][] emptyArray = Array.Empty<double[][]>();
            int actionCount = 0;
            Action<double> countAction = (x) => actionCount++;

            // Act & Assert
            var exception = Record.Exception(() => NetworkCreator.ActOn3dArr(emptyArray, countAction));
            Assert.Null(exception);
            Assert.Equal(0, actionCount);
        }

        [Fact]
        public void ApplyOn3dArr_TransformsAllElements()
        {
            // Arrange
            double[][][] array = {
                new double[][] { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } },
                new double[][] { new double[] { 5.0, 6.0 } }
            };
            Func<double, double> doubleFunc = x => x * 2.0;

            // Act
            NetworkCreator.ApplyOn3dArr(array, doubleFunc);

            // Assert
            Assert.Equal(2.0, array[0][0][0]);
            Assert.Equal(4.0, array[0][0][1]);
            Assert.Equal(6.0, array[0][1][0]);
            Assert.Equal(8.0, array[0][1][1]);
            Assert.Equal(10.0, array[1][0][0]);
            Assert.Equal(12.0, array[1][0][1]);
        }

        [Fact]
        public void ApplyOn3dArr_WithIdentityFunction_LeavesElementsUnchanged()
        {
            // Arrange
            double[][][] array = {
                new double[][] { new double[] { 1.0, 2.0 } }
            };
            var originalValues = new double[] { array[0][0][0], array[0][0][1] };
            Func<double, double> identityFunc = x => x;

            // Act
            NetworkCreator.ApplyOn3dArr(array, identityFunc);

            // Assert
            Assert.Equal(originalValues[0], array[0][0][0]);
            Assert.Equal(originalValues[1], array[0][0][1]);
        }

        [Fact]
        public void ApplyOn3dArr_WithEmptyArray_DoesNotThrow()
        {
            // Arrange
            double[][][] emptyArray = Array.Empty<double[][]>();
            Func<double, double> doubleFunc = x => x * 2.0;

            // Act & Assert
            var exception = Record.Exception(() => NetworkCreator.ApplyOn3dArr(emptyArray, doubleFunc));
            Assert.Null(exception);
        }

        #endregion

        #region CreateNetwork Tests

        [Fact]
        public void CreateNetwork_ReturnsNeuralNetworkInstance()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Sigmoid };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.NotNull(network);
            Assert.IsAssignableFrom<INeuralNetwork>(network);
        }

        [Fact]
        public void CreateNetwork_PassesCorrectParametersToNetwork()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Sigmoid, ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            creator.RandomizeWeights(-1.0, 1.0);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.Equal(creator.Weights, network.Weigths);
            Assert.Equal(creator.Biases, network.Biases);
            Assert.Equal(creator.Ys, network.Ys);
            Assert.Equal(creator.ActivationFunctions, network.ActivationFunctions);
        }

        [Fact]
        public void CreateNetwork_CreatesWorkingNetwork()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit, ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            creator.RandomizeWeights(-1.0, 1.0);

            // Act
            var network = creator.CreateNetwork();
            var prediction = network.Predict(new double[] { 1.0, 2.0 });

            // Assert
            Assert.NotNull(prediction);
            Assert.Single(prediction);
            Assert.True(double.IsFinite(prediction[0]));
        }

        [Fact]
        public void CreateNetwork_WithEmptyConfiguration_CreatesEmptyNetwork()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = Array.Empty<int>();
            Func<double, double>[] activationFunctions = Array.Empty<Func<double, double>>();
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.NotNull(network);
            Assert.Empty(network.Layers);
        }

        #endregion

        #region Edge Case Tests

        [Fact]
        public void NetworkCreator_Constructor_WithNullActivationFunctions_DoesNotThrow()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 1 };

            // Act & Assert
            var exception = Record.Exception(() => new NetworkCreator(inputs, layerSizes, null!));
            Assert.Null(exception);
        }

        [Fact]
        public void NetworkCreator_Constructor_WithNullLayerSizes_ThrowsException()
        {
            // Arrange
            int inputs = 2;
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act & Assert
            Assert.Throws<NullReferenceException>(() => new NetworkCreator(inputs, null!, activationFunctions));
        }

        [Fact]
        public void NetworkCreator_Constructor_WithVeryLargeLayerSizes_HandlesCorrectly()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 1000 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Assert
            Assert.Single(creator.Weights);
            Assert.Equal(1000, creator.Weights[0].Length);
            Assert.Equal(2, creator.Weights[0][0].Length);
        }

        [Fact]
        public void NetworkCreator_Properties_CanBeModifiedAfterConstruction()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

            // Act
            creator.Weights[0][0][0] = 5.0;
            creator.Biases[0][0][0] = 3.0;

            // Assert
            Assert.Equal(5.0, creator.Weights[0][0][0]);
            Assert.Equal(3.0, creator.Biases[0][0][0]);
        }

        #endregion

        #region Integration Tests

        [Fact]
        public void NetworkCreator_CompleteWorkflow_CreatesTrainableNetwork()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 3, 2, 1 };
            Func<double, double>[] activationFunctions = { 
                ActivationFunctions.Sigmoid, 
                ActivationFunctions.Sigmoid, 
                ActivationFunctions.Sigmoid 
            };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            creator.RandomizeWeights(-1.0, 1.0);

            // Act
            var network = creator.CreateNetwork();
            var trainer = new NeuralNetworkTrainer(network, 0.1);
            
            double[][] trainingData = { new double[] { 1.0, 2.0 } };
            double[][] observed = { new double[] { 3.0 } };
            
            var initialPrediction = network.Predict(trainingData[0]);
            trainer.TrainOneEpoch(trainingData, observed);
            var finalPrediction = network.Predict(trainingData[0]);

            // Assert
            Assert.NotNull(network);
            Assert.NotNull(trainer);
            Assert.NotEqual(initialPrediction[0], finalPrediction[0]);
        }

        [Fact]
        public void NetworkCreator_MultipleNetworksFromSameCreator_AreIndependent()
        {
            // Arrange
            int inputs = 2;
            int[] layerSizes = { 2, 1 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit, ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
            creator.RandomizeWeights(-1.0, 1.0);

            // Act
            var network1 = creator.CreateNetwork();
            var network2 = creator.CreateNetwork();
            
            // Modify network1 weights
            network1.Weigths[0][0][0] = 999.0;

            // Assert
            Assert.NotEqual(network1.Weigths[0][0][0], network2.Weigths[0][0][0]);
        }

        #endregion
    }
}
