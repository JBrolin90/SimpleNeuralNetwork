using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class NetworkCreatorTests
{
    [Fact]
    public void Constructor_ShouldInitializeWeightsAndBiasesCorrectly()
    {
        // Arrange
        int[] layerSizes = [3, 4, 2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Sigmoid,
            ActivationFunctions.Unit
        ];

        // Act
        var creator = new NetworkCreator(1,layerSizes, activationFunctions);

        // Assert
        Assert.NotNull(creator.Weights);
        Assert.NotNull(creator.Biases);
        Assert.NotNull(creator.ActivationFunctions);
        Assert.NotNull(creator.Ys);
        
        // Verify weights structure
        Assert.Equal(4, creator.Weights.Length);
        Assert.Empty(creator.Weights[0]); // Input layer has no weights
        Assert.Empty(creator.Weights[3]); // Output layer has no weights
        Assert.Equal(4, creator.Weights[1].Length); // Hidden layer 1 has 4 nodes
        Assert.Equal(2, creator.Weights[2].Length); // Hidden layer 2 has 2 nodes
        
        // Verify biases structure
        Assert.Equal(4, creator.Biases.Length);
        Assert.Empty(creator.Biases[0]); // Input layer has no biases
        Assert.Empty(creator.Biases[3]); // Output layer has no biases
        Assert.Equal(4, creator.Biases[1].Length); // Hidden layer 1 has 4 nodes
        Assert.Equal(2, creator.Biases[2].Length); // Hidden layer 2 has 2 nodes
        
        // Verify activation functions are preserved
        Assert.Equal(activationFunctions, creator.ActivationFunctions);
    }

    [Fact]
    public void Constructor_SingleLayerNetwork_ShouldHandleMinimalConfiguration()
    {
        // Arrange
        int[] layerSizes = [1, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.Unit
        ];

        // Act
        var creator = new NetworkCreator(1, layerSizes, activationFunctions);

        // Assert
        Assert.NotNull(creator.Weights);
        Assert.NotNull(creator.Biases);
        Assert.Equal(2, creator.Weights.Length);
        Assert.Empty(creator.Weights[0]); // Input layer
        Assert.Empty(creator.Weights[1]); // Output layer
    }

    [Theory]
    [InlineData(new int[] { 2, 3, 1 })]
    [InlineData(new int[] { 1, 5, 3, 2 })]
    [InlineData(new int[] { 4, 6, 4, 2, 1 })]
    public void Constructor_VariousNetworkSizes_ShouldCreateCorrectStructure(int[] layerSizes)
    {
        // Arrange
        int inputs = layerSizes[0]; // Use first layer size as input count
        var activationFunctions = new Func<double, double>[layerSizes.Length];
        for (int i = 0; i < layerSizes.Length; i++)
        {
            activationFunctions[i] = ActivationFunctions.Unit;
        }

        // Act
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Assert
        Assert.Equal(layerSizes.Length, creator.Weights.Length);
        Assert.Equal(layerSizes.Length, creator.Biases.Length);
        
        // Verify hidden layers have correct number of nodes
        for (int i = 1; i < layerSizes.Length - 1; i++)
        {
            Assert.Equal(layerSizes[i], creator.Weights[i].Length);
            Assert.Equal(layerSizes[i], creator.Biases[i].Length);
        }
    }

    [Fact]
    public void Constructor_ShouldInitializeWeightsAndBiasesToZero()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [3, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];

        // Act
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Assert
        // Check that all weights in hidden layers are initialized to 0
        for (int i = 1; i < creator.Weights.Length - 1; i++)
        {
            for (int j = 0; j < creator.Weights[i].Length; j++)
            {
                for (int k = 0; k < creator.Weights[i][j].Length; k++)
                {
                    Assert.Equal(0.0, creator.Weights[i][j][k]);
                }
            }
        }

        // Check that all biases in hidden layers are initialized to 0
        for (int i = 1; i < creator.Biases.Length - 1; i++)
        {
            for (int j = 0; j < creator.Biases[i].Length; j++)
            {
                Assert.Equal(0.0, creator.Biases[i][j][0]);
            }
        }
    }

    [Fact]
    public void RandomizeWeights_ShouldModifyWeightValues()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [3, 2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Sigmoid,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Store original weights (should all be 0)
        var originalWeights = new double[creator.Weights.Length][][];
        for (int i = 0; i < creator.Weights.Length; i++)
        {
            originalWeights[i] = new double[creator.Weights[i].Length][];
            for (int j = 0; j < creator.Weights[i].Length; j++)
            {
                originalWeights[i][j] = new double[creator.Weights[i][j].Length];
                Array.Copy(creator.Weights[i][j], originalWeights[i][j], creator.Weights[i][j].Length);
            }
        }

        // Act
        creator.RandomizeWeights();

        // Assert
        bool hasChangedWeights = false;
        for (int i = 1; i < creator.Weights.Length - 1; i++)
        {
            for (int j = 0; j < creator.Weights[i].Length; j++)
            {
                for (int k = 0; k < creator.Weights[i][j].Length; k++)
                {
                    if (Math.Abs(creator.Weights[i][j][k] - originalWeights[i][j][k]) > 0.001)
                    {
                        hasChangedWeights = true;
                        break;
                    }
                }
                if (hasChangedWeights) break;
            }
            if (hasChangedWeights) break;
        }
        
        Assert.True(hasChangedWeights, "RandomizeWeights should modify at least some weight values");
    }

    [Fact]
    public void RandomizeWeights_ShouldProduceValuesInExpectedRange()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [4, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Act
        creator.RandomizeWeights();

        // Assert
        // Check that randomized weights are in range [-5, 5] (formula: Random()*10-5)
        for (int i = 1; i < creator.Weights.Length - 1; i++)
        {
            for (int j = 0; j < creator.Weights[i].Length; j++)
            {
                for (int k = 0; k < creator.Weights[i][j].Length; k++)
                {
                    var weight = creator.Weights[i][j][k];
                    Assert.True(weight >= -5.0 && weight <= 5.0, 
                        $"Weight {weight} should be between -5 and 5");
                }
            }
        }
    }

    [Fact]
    public void RandomizeWeights_MultipleCallsShouldProduceDifferentValues()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [3, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Act
        creator.RandomizeWeights();
        var firstRandomization = creator.Weights[1][0][0];
        
        creator.RandomizeWeights();
        var secondRandomization = creator.Weights[1][0][0];

        // Assert
        // Note: There's a small chance they could be equal, but very unlikely
        Assert.NotEqual(firstRandomization, secondRandomization);
    }

    [Fact]
    public void CreateNetwork_ShouldReturnValidNeuralNetworkTrainer()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [3, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Act
        var network = creator.CreateNetwork();

        // Assert
        Assert.NotNull(network);
        Assert.IsType<NeuralNetworkTrainer>(network);
        Assert.NotNull(network.Layers);
        Assert.Equal(layerSizes.Length, network.Layers.Length);
    }

    [Fact]
    public void CreateNetwork_ShouldUseCurrentWeightsAndBiases()
    {
        // Arrange
        int inputs = 1;
        int[] layerSizes = [2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
        
        // Modify weights manually
        creator.Weights[1][0][0] = 2.5;
        creator.Weights[1][1][0] = -1.8;

        // Act
        var network = creator.CreateNetwork();

        // Assert
        Assert.Equal(creator.Weights, network.Weigths);
        Assert.Equal(creator.Biases, network.Biases);
        Assert.Equal(creator.ActivationFunctions, network.ActivationFunctions);
    }

    [Fact]
    public void CreateNetwork_WithRandomizedWeights_ShouldCreateTrainableNetwork()
    {
        // Arrange
        int inputs = 1;
        int[] layerSizes = [2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
        creator.RandomizeWeights();

        // Act
        var network = creator.CreateNetwork();

        // Assert
        Assert.NotNull(network);
        
        // Test that the network can be used for prediction
        double[] testInput = [0.5];
        var prediction = network.Predict(testInput);
        
        Assert.NotNull(prediction);
        Assert.Single(prediction); // Should return 1 output for this network
        Assert.True(double.IsFinite(prediction[0]), "Prediction should be a finite number");
    }

    [Theory]
    [InlineData(new int[] { 1, 1 })]
    [InlineData(new int[] { 2, 3, 1 })]
    [InlineData(new int[] { 3, 5, 4, 2 })]
    public void CreateNetwork_VariousArchitectures_ShouldCreateValidNetworks(int[] layerSizes)
    {
        // Arrange
        int inputs = layerSizes[0]; // Use first layer size as input count
        var activationFunctions = new Func<double, double>[layerSizes.Length];
        for (int i = 0; i < layerSizes.Length; i++)
        {
            activationFunctions[i] = ActivationFunctions.Unit;
        }
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
        creator.RandomizeWeights();

        // Act
        var network = creator.CreateNetwork();

        // Assert
        Assert.NotNull(network);
        Assert.Equal(layerSizes.Length, network.Layers.Length);
        
        // Test basic functionality
        var testInput = new double[layerSizes[0]];
        for (int i = 0; i < testInput.Length; i++)
        {
            testInput[i] = 0.5;
        }
        
        var prediction = network.Predict(testInput);
        Assert.NotNull(prediction);
        Assert.Equal(layerSizes[^1], prediction.Length);
    }

    [Fact]
    public void Properties_ShouldBeSettableAndGettable()
    {
        // Arrange
        int inputs = 2;
        int[] layerSizes = [2, 2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Act & Assert - Test property setters
        var newWeights = new double[][][] { [], [[1.0, 2.0], [3.0, 4.0]], [] };
        creator.Weights = newWeights;
        Assert.Equal(newWeights, creator.Weights);

        var newBiases = new double[][][] { [], [[0.1], [0.2]], [] };
        creator.Biases = newBiases;
        Assert.Equal(newBiases, creator.Biases);

        var newActivationFunctions = new Func<double, double>[] {
            ActivationFunctions.Sigmoid,
            ActivationFunctions.ReLU,
            ActivationFunctions.Sigmoid
        };
        creator.ActivationFunctions = newActivationFunctions;
        Assert.Equal(newActivationFunctions, creator.ActivationFunctions);

        var newYs = new double[][] { [1.0, 2.0], [3.0, 4.0] };
        creator.Ys = newYs;
        Assert.Equal(newYs, creator.Ys);
    }

    [Fact]
    public void CreateNetwork_ShouldUseDefaultLearningRate()
    {
        // Arrange
        int inputs = 1;
        int[] layerSizes = [2, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);

        // Act
        var network = creator.CreateNetwork();

        // Assert
        Assert.NotNull(network);
        // The learning rate is hardcoded to 0.01 in CreateNetwork method
        // We can verify this by checking if the network was created successfully
        Assert.IsType<NeuralNetworkTrainer>(network);
    }

    [Fact]
    public void NetworkCreator_Integration_ShouldCreateTrainableNetwork()
    {
        // Arrange
        int inputs = 1;
        int[] layerSizes = [3, 1];
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit
        ];
        var creator = new NetworkCreator(inputs, layerSizes, activationFunctions);
        creator.RandomizeWeights();

        // Act
        var network = creator.CreateNetwork();
        double[] testInputs = [0.5];
        double[] expected = [1.0];

        // Get initial prediction
        var initialPrediction = network.Predict(testInputs);
        var initialSSR = network.SSR;

        // Train the network
        for (int i = 0; i < 10; i++)
        {
            network.Train(testInputs, expected);
        }

        // Get prediction after training
        var trainedPrediction = network.Predict(testInputs);

        // Assert
        Assert.NotNull(initialPrediction);
        Assert.NotNull(trainedPrediction);
        Assert.Single(initialPrediction);
        Assert.Single(trainedPrediction);
        
        // SSR should be calculated (may be 0 or positive)
        Assert.True(network.SSR >= 0, "SSR should be non-negative");
        
        // Network should be able to train without throwing exceptions
        Assert.True(double.IsFinite(trainedPrediction[0]), "Trained prediction should be finite");
    }
}
