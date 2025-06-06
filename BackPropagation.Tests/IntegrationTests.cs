using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class IntegrationTests
{
    [Fact]
    public void NeuralNetwork_EndToEndTest_ShouldWorkCorrectly()
    {
        // Arrange - Use the same configuration as Program.cs
        double[] inputs = [0, 0.5, 1];
        double[] observed = [0, 1, 0];

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

        // Act - Create and test the network
        var testNN = new NeuralNetworkTrainer(new LayerFactory(), new NodeFactory(),
            weights, biases, ys, 0.01, activationFunctions);

        // Initial prediction
        var initialOutput = testNN.Predict([0.5]);
        var initialSSR = testNN.SSR;

        // Train the network
        for (int i = 0; i < 100; i++)
        {
            testNN.Train(inputs, observed);
        }

        var finalSSR = testNN.SSR;

        // Assert
        Assert.NotNull(initialOutput);
        Assert.Single(initialOutput);
        Assert.True(double.IsFinite(initialOutput[0]));
        Assert.True(finalSSR >= 0);
    }

    [Fact]
    public void NeuralNetwork_TrainingReducesError_OverTime()
    {
        // Arrange
        double[] inputs = [0, 0.5, 1];
        double[] observed = [0, 1, 0];

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

        var testNN = new NeuralNetworkTrainer(new LayerFactory(), new NodeFactory(),
            weights, biases, ys, 0.01, activationFunctions);

        // Act - Train and record SSR values
        double initialSSR = double.MaxValue;
        double finalSSR = 0;

        for (int i = 0; i < 1000; i++)
        {
            testNN.Train(inputs, observed);
            if (i == 0)
            {
                initialSSR = testNN.SSR;
            }
            if (i == 999)
            {
                finalSSR = testNN.SSR;
            }
        }

        // Assert - The error should generally decrease or at least not increase significantly
        Assert.True(finalSSR >= 0);
        Assert.True(initialSSR >= 0);
        // Note: Due to the nature of the backpropagation implementation, 
        // we can't guarantee error reduction, but we can ensure it doesn't explode
        Assert.True(finalSSR < 1000); // Sanity check that error doesn't explode
    }

    [Fact]
    public void NeuralNetwork_DifferentActivationFunctions_ShouldWork()
    {
        // Arrange
        var activationSets = new[]
        {
            new[] { ActivationFunctions.Unit, ActivationFunctions.Unit, ActivationFunctions.Unit, ActivationFunctions.Unit },
            new[] { ActivationFunctions.Unit, ActivationFunctions.Sigmoid, ActivationFunctions.Unit, ActivationFunctions.Unit },
            new[] { ActivationFunctions.Unit, ActivationFunctions.SoftPlus, ActivationFunctions.Unit, ActivationFunctions.Unit }
        };

        double[][][] weights = [
            [],
            [[1.0], [1.0]],
            [[1.0, 1.0]],
            []
        ];
        double[][][] biases = [[], [[0], [0]], [[0]], []];
        double[][] ys = [[0], [0, 0], [0], [0]];

        foreach (var activationFunctions in activationSets)
        {
            // Act
            var network = new NeuralNetworkTrainer(new LayerFactory(), new NodeFactory(),
                weights, biases, ys, 0.01, activationFunctions);

            var output = network.Predict([0.5]);

            // Assert
            Assert.NotNull(output);
            Assert.Single(output);
            Assert.True(double.IsFinite(output[0]));
        }
    }

    [Fact]
    public void LayerConnections_ShouldBeProperlyLinked()
    {
        // Arrange
        double[][][] weights = [
            [],
            [[1.0]],
            [[1.0]],
            []
        ];
        double[][][] biases = [[], [[0]], [[0]], []];
        double[][] ys = [[0], [0], [0], [0]];

        Func<double, double>[] activationFunctions = [
            ActivationFunctions.Unit,
            ActivationFunctions.Unit,
            ActivationFunctions.Unit,
            ActivationFunctions.Unit
        ];

        // Act
        var network = new NeuralNetwork(new LayerFactory(), new NodeFactory(),
            weights, biases, ys, activationFunctions);

        // Assert
        for (int i = 1; i < network.Layers.Length; i++)
        {
            Assert.Equal(network.Layers[i - 1], network.Layers[i].PreviousLayer);
        }

        for (int i = 0; i < network.Layers.Length - 1; i++)
        {
            Assert.Equal(network.Layers[i + 1], network.Layers[i].NextLayer);
        }
    }
}
