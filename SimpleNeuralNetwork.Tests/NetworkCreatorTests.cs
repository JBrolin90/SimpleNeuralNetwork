using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class NetworkCreatorTests
    {
        private const double Tolerance = 1e-7;

        [Fact]
        public void NetworkCreator_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int inputSize = 3;
            int[] layerSizes = { 5, 2 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid,
                ActivationFunctions.Unit
            };

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            Assert.NotNull(creator);
            Assert.NotNull(creator.Weights);
            Assert.NotNull(creator.Biases);
        }

        [Fact]
        public void NetworkCreator_Constructor_CreatesCorrectWeightDimensions()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 3, 1 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(2, creator.Weights.Length); // 2 layers
            
            // First layer: 3 nodes, each with 2 inputs
            Assert.Equal(3, creator.Weights[0].Length);
            Assert.Equal(2, creator.Weights[0][0].Length);
            Assert.Equal(2, creator.Weights[0][1].Length);
            Assert.Equal(2, creator.Weights[0][2].Length);
            
            // Second layer: 1 node, with 3 inputs (from previous layer)
            Assert.Single(creator.Weights[1]);
            Assert.Equal(3, creator.Weights[1][0].Length);
        }

        [Fact]
        public void NetworkCreator_Constructor_CreatesCorrectBiasDimensions()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 3, 1 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(2, creator.Biases.Length); // 2 layers
            
            // First layer: 3 nodes, each with 1 bias
            Assert.Equal(3, creator.Biases[0].Length);
            Assert.Single(creator.Biases[0][0]);
            Assert.Single(creator.Biases[0][1]);
            Assert.Single(creator.Biases[0][2]);
            
            // Second layer: 1 node, with 1 bias
            Assert.Single(creator.Biases[1]);
            Assert.Single(creator.Biases[1][0]);
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesWeightsToZero()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            foreach (var layer in creator.Weights)
            {
                foreach (var node in layer)
                {
                    foreach (var weight in node)
                    {
                        Assert.Equal(0.0, weight, 7);
                    }
                }
            }
        }

        [Fact]
        public void NetworkCreator_Constructor_InitializesBiasesToZero()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            foreach (var layer in creator.Biases)
            {
                foreach (var node in layer)
                {
                    foreach (var bias in node)
                    {
                        Assert.Equal(0.0, bias, 7);
                    }
                }
            }
        }

        [Fact]
        public void NetworkCreator_RandomizeWeights_ChangesWeightValues()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Store original weights (should be 0)
            var originalWeights = creator.Weights[0][0][0];

            // Act
            creator.RandomizeWeights();

            // Assert
            Assert.NotEqual(originalWeights, creator.Weights[0][0][0]);
        }

        [Fact]
        public void NetworkCreator_RandomizeWeights_WithMultipleCalls_ProducesDifferentResults()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Act
            creator.RandomizeWeights();
            var firstRandomWeight = creator.Weights[0][0][0];
            
            creator.RandomizeWeights();
            var secondRandomWeight = creator.Weights[0][0][0];

            // Assert
            // Note: There's a very small chance these could be equal, but it's extremely unlikely
            Assert.NotEqual(firstRandomWeight, secondRandomWeight);
        }

        [Fact]
        public void NetworkCreator_RandomizeWeights_ProducesNonZeroValues()
        {
            // Arrange
            int inputSize = 3;
            int[] layerSizes = { 4, 2 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Act
            creator.RandomizeWeights();

            // Assert
            bool hasNonZeroWeight = false;
            foreach (var layer in creator.Weights)
            {
                foreach (var node in layer)
                {
                    foreach (var weight in node)
                    {
                        if (Math.Abs(weight) > 1e-10) // Check if weight is significantly different from zero
                        {
                            hasNonZeroWeight = true;
                        }
                        // Just verify that weights are finite numbers
                        Assert.True(double.IsFinite(weight), $"Weight {weight} should be a finite number");
                    }
                }
            }
            Assert.True(hasNonZeroWeight, "At least some weights should be non-zero after randomization");
        }

        [Fact]
        public void NetworkCreator_CreateNetwork_ReturnsValidNeuralNetworkTrainer()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 3, 1 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Sigmoid,
                ActivationFunctions.Unit
            };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.NotNull(network);
            Assert.IsType<NeuralNetworkTrainer>(network);
            Assert.Equal(2, network.Layers.Length);
        }

        [Fact]
        public void NetworkCreator_CreateNetwork_SetsDefaultLearningRate()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 2 };
            Func<double, double>[] activationFunctions = { ActivationFunctions.Unit };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.Equal(0.01, network.LearningRate); // Default learning rate
        }

        [Theory]
        [InlineData(1, new int[] { 1 })]
        [InlineData(3, new int[] { 5, 3, 1 })]
        [InlineData(2, new int[] { 4, 4, 2 })]
        public void NetworkCreator_WithVariousConfigurations_CreatesValidStructure(int inputSize, int[] layerSizes)
        {
            // Arrange
            var activationFunctions = new Func<double, double>[layerSizes.Length];
            for (int i = 0; i < layerSizes.Length; i++)
            {
                activationFunctions[i] = ActivationFunctions.Unit;
            }

            // Act
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Assert
            Assert.Equal(layerSizes.Length, creator.Weights.Length);
            Assert.Equal(layerSizes.Length, creator.Biases.Length);
            
            for (int i = 0; i < layerSizes.Length; i++)
            {
                Assert.Equal(layerSizes[i], creator.Weights[i].Length);
                Assert.Equal(layerSizes[i], creator.Biases[i].Length);
                
                int expectedInputs = i == 0 ? inputSize : layerSizes[i - 1];
                foreach (var nodeWeights in creator.Weights[i])
                {
                    Assert.Equal(expectedInputs, nodeWeights.Length);
                }
            }
        }

        [Fact]
        public void NetworkCreator_CreateYs_CreatesCorrectDimensions()
        {
            // Arrange
            int inputSize = 2;
            int[] layerSizes = { 3, 2 };
            Func<double, double>[] activationFunctions = {
                ActivationFunctions.Unit,
                ActivationFunctions.Unit
            };
            var creator = new NetworkCreator(inputSize, layerSizes, activationFunctions);

            // Act
            var network = creator.CreateNetwork();

            // Assert
            Assert.Equal(2, network.Ys.Length); // 2 layers
            Assert.Equal(3, network.Ys[0].Length); // First layer has 3 nodes
            Assert.Equal(2, network.Ys[1].Length); // Second layer has 2 nodes
        }
    }
}
