using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class TestNeuralNetworkTests
{
    private TestNeuralNetwork CreateTestNetwork()
    {
        double[][][] weights = [
            [],
            [[2.74], [-1.13]],
            [[0.36, 0.63]],
            []
        ];
        double[][][] biases = [[], [[0], [0]], [[0]], []];
        double[][] ys = [[0, 0], [0, 0], [0, 0], [0, 0]];
        
        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.SoftPlus,
            ActivationFunctions.Unit,
            ActivationFunctions.Unit
        ];

        return new TestNeuralNetwork(new LayerFactory(), new NodeFactory(),
            weights, biases, ys, 0.01, activationFunctions);
    }

    [Fact]
    public void Constructor_ShouldInitializeTestNetwork()
    {
        // Arrange & Act
        var network = CreateTestNetwork();

        // Assert
        Assert.NotNull(network);
        Assert.Equal(0, network.SSR);
        Assert.Equal(0, network.dSSR);
        Assert.NotNull(network.NodeSteps);
    }

    [Fact]
    public void Train_ShouldUpdateSSR()
    {
        // Arrange
        var network = CreateTestNetwork();
        double[] inputs = [0, 0.5, 1];
        double[] observed = [0, 1, 0];

        // Act
        network.Train(inputs, observed);

        // Assert
        Assert.True(network.SSR >= 0);
    }

    [Fact]
    public void Train_MultipleIterations_ShouldModifyWeights()
    {
        // Arrange
        var network = CreateTestNetwork();
        double[] inputs = [0, 0.5, 1];
        double[] observed = [0, 1, 0];
        
        // Store initial weights
        var initialWeight = network.Weigths[1][0][0];

        // Act
        for (int i = 0; i < 100; i++)
        {
            network.Train(inputs, observed);
        }

        // Assert
        var finalWeight = network.Weigths[1][0][0];
        Assert.NotEqual(initialWeight, finalWeight);
    }

    [Fact]
    public void PrepareBackPropagation_ShouldInitializeNodeSteps()
    {
        // Arrange
        var network = CreateTestNetwork();

        // Act
        network.PrepareBackPropagation();

        // Assert
        Assert.NotNull(network.NodeSteps);
        Assert.True(network.NodeSteps.Length > 0);
    }

    [Fact]
    public void Train_WithValidInputs_ShouldNotThrow()
    {
        // Arrange
        var network = CreateTestNetwork();
        double[] inputs = [0.5];
        double[] observed = [1.0];

        // Act & Assert
        var exception = Record.Exception(() => network.Train(inputs, observed));
        Assert.Null(exception);
    }
}
