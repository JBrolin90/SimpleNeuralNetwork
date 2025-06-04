using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class NodeTests
{
    private Node CreateTestNode()
    {
        var nodeFactory = new NodeFactory();
        var layer = new Layer(0, nodeFactory, [[1.0, 2.0]], [[0.1]], ActivationFunctions.Unit);
        double[] weights = [1.0, 2.0];
        double[] bias = [0.1];
        
        return new Node(layer, 0, weights, bias, ActivationFunctions.Unit);
    }

    [Fact]
    public void Constructor_ShouldInitializeNode()
    {
        // Arrange & Act
        var node = CreateTestNode();

        // Assert
        Assert.NotNull(node);
        Assert.NotNull(node.Weights);
        Assert.NotNull(node.Bias);
        Assert.Equal(2, node.Weights.Length);
        Assert.Single(node.Bias);
        Assert.Equal(ActivationFunctions.Unit, node.ActivationFunction);
        Assert.Equal(ActivationFunctions.UnitDerivative, node.ActivationDerivative);
    }

    [Fact]
    public void ProcessInputs_ShouldCalculateOutput()
    {
        // Arrange
        var node = CreateTestNode();
        double[] inputs = [0.5, 1.5];

        // Act
        var output = node.ProcessInputs(inputs);

        // Assert
        Assert.True(double.IsFinite(output));
        // Expected: (0.5 * 1.0) + (1.5 * 2.0) + 0.1 = 0.5 + 3.0 + 0.1 = 3.6
        Assert.Equal(3.6, output, 0.001);
    }

    [Theory]
    [InlineData(new double[] { 1.0, 0.0 }, 1.1)] // 1*1 + 0*2 + 0.1 = 1.1
    [InlineData(new double[] { 0.0, 1.0 }, 2.1)] // 0*1 + 1*2 + 0.1 = 2.1
    [InlineData(new double[] { 0.0, 0.0 }, 0.1)] // 0*1 + 0*2 + 0.1 = 0.1
    public void ProcessInputs_WithDifferentInputs_ShouldReturnExpectedOutput(double[] inputs, double expected)
    {
        // Arrange
        var node = CreateTestNode();

        // Act
        var output = node.ProcessInputs(inputs);

        // Assert
        Assert.Equal(expected, output, 0.001);
    }

    [Fact]
    public void Constructor_WithSigmoidActivation_ShouldSetCorrectDerivative()
    {
        // Arrange
        var nodeFactory = new NodeFactory();
        var layer = new Layer(0, nodeFactory, [[1.0]], [[0.0]], ActivationFunctions.Sigmoid);
        double[] weights = [1.0];
        double[] bias = [0.0];

        // Act
        var node = new Node(layer, 0, weights, bias, ActivationFunctions.Sigmoid);

        // Assert
        Assert.Equal(ActivationFunctions.Sigmoid, node.ActivationFunction);
        Assert.Equal(ActivationFunctions.SigmoidDerivative, node.ActivationDerivative);
    }

    [Fact]
    public void Constructor_WithSoftPlusActivation_ShouldSetCorrectDerivative()
    {
        // Arrange
        var nodeFactory = new NodeFactory();
        var layer = new Layer(0, nodeFactory, [[1.0]], [[0.0]], ActivationFunctions.SoftPlus);
        double[] weights = [1.0];
        double[] bias = [0.0];

        // Act
        var node = new Node(layer, 0, weights, bias, ActivationFunctions.SoftPlus);

        // Assert
        Assert.Equal(ActivationFunctions.SoftPlus, node.ActivationFunction);
        Assert.Equal(ActivationFunctions.SoftPlusDerivative, node.ActivationDerivative);
    }

    [Fact]
    public void BiasDerivative_ShouldReturnOne()
    {
        // Arrange
        var node = CreateTestNode();

        // Act
        var derivative = node.BiasDerivative();

        // Assert
        Assert.Equal(1.0, derivative);
    }
}
