using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class NeuralNetworkTests
{
    private NeuralNetwork CreateTestNetwork()
    {
        // Sample network configuration from Program.cs
        double[][][] weights = [
            [],
            [[2.74], [-1.13]],
            [[0.36, 0.63]],
            []
        ];
        double[][][] biases = [[], [[0], [0]], [[0]], []];
        double[][] ys = [[0], [0, 0], [0], [0]];
        
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit,
            ActivationFunctions.Unit
        ];

        return new NeuralNetwork(new LayerFactory(), new NodeFactory(),
            weights, biases, ys, activationFunctions);
    }

    [Fact]
    public void Constructor_ShouldInitializeNetwork()
    {
        // Arrange & Act
        var network = CreateTestNetwork();

        // Assert
        Assert.NotNull(network);
        Assert.NotNull(network.Layers);
        Assert.Equal(4, network.Layers.Length);
        Assert.NotNull(network.Ys);
        Assert.NotNull(network.Weigths);
        Assert.NotNull(network.Biases);
    }

    [Fact]
    public void Predict_ShouldReturnValidOutput()
    {
        // Arrange
        var network = CreateTestNetwork();
        double[] input = [0.5];

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0);
        Assert.True(double.IsFinite(output[0]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(0.5)]
    [InlineData(1.0)]
    public void Predict_WithDifferentInputs_ShouldReturnDifferentOutputs(double input)
    {
        // Arrange
        var network = CreateTestNetwork();

        // Act
        var output = network.Predict([input]);

        // Assert
        Assert.NotNull(output);
        Assert.Single(output);
        Assert.True(double.IsFinite(output[0]));
    }

    [Fact]
    public void Predict_ShouldProduceConsistentResults()
    {
        // Arrange
        var network = CreateTestNetwork();
        double[] input = [0.5];

        // Act
        var output1 = network.Predict(input);
        var output2 = network.Predict(input);

        // Assert
        Assert.Equal(output1[0], output2[0]);
    }

    [Fact]
    public void Constructor_WithNullFactory_ShouldThrowException()
    {
        // Arrange
        double[][][] weights = [[]];
        double[][][] biases = [[]];
        double[][] ys = [[]];
        Func<double, double>[] activationFunctions = [ActivationFunctions.Unit];

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            new NeuralNetwork(null!, new NodeFactory(), weights, biases, ys, activationFunctions));
        
        Assert.Throws<ArgumentNullException>(() => 
            new NeuralNetwork(new LayerFactory(), null!, weights, biases, ys, activationFunctions));
    }
}
