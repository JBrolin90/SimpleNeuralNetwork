using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class LayerTests
{
    private Layer CreateTestLayer()
    {
        var nodeFactory = new NodeFactory();
        double[][] weights = [[1.0, 2.0], [0.5, 1.5]];
        double[][] biases = [[0.1], [0.2]];
        
        return new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);
    }

    [Fact]
    public void Constructor_ShouldInitializeLayer()
    {
        // Arrange & Act
        var layer = CreateTestLayer();

        // Assert
        Assert.NotNull(layer);
        Assert.NotNull(layer.Nodes);
        Assert.Equal(2, layer.Nodes.Length);
        Assert.Equal(0, layer.Index);
    }

    [Fact]
    public void Forward_ShouldProcessInputs()
    {
        // Arrange
        var layer = CreateTestLayer();
        double[] inputs = [1.0, 2.0];

        // Act
        var outputs = layer.Forward(inputs);

        // Assert
        Assert.NotNull(outputs);
        Assert.Equal(2, outputs.Length);
        Assert.All(outputs, output => Assert.True(double.IsFinite(output)));
    }

    [Fact]
    public void Forward_WithEmptyInputs_ShouldHandleGracefully()
    {
        // Arrange
        var nodeFactory = new NodeFactory();
        double[][] weights = [[]];
        double[][] biases = [[0.0]];
        var layer = new Layer(0, nodeFactory, weights, biases, ActivationFunctions.Unit);

        // Act
        var outputs = layer.Forward([]);

        // Assert
        Assert.NotNull(outputs);
    }

    [Fact]
    public void SetPreviousLayer_ShouldSetCorrectly()
    {
        // Arrange
        var layer1 = CreateTestLayer();
        var layer2 = CreateTestLayer();

        // Act
        layer2.PreviousLayer = layer1;

        // Assert
        Assert.Equal(layer1, layer2.PreviousLayer);
    }

    [Fact]
    public void SetNextLayer_ShouldSetCorrectly()
    {
        // Arrange
        var layer1 = CreateTestLayer();
        var layer2 = CreateTestLayer();

        // Act
        layer1.NextLayer = layer2;

        // Assert
        Assert.Equal(layer2, layer1.NextLayer);
    }

    [Fact]
    public void PreviousLayer_WhenNotSet_ShouldThrowException()
    {
        // Arrange
        var layer = CreateTestLayer();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => layer.PreviousLayer);
    }

    [Fact]
    public void NextLayer_WhenNotSet_ShouldThrowException()
    {
        // Arrange
        var layer = CreateTestLayer();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => layer.NextLayer);
    }

    [Fact]
    public void SetPreviousLayer_WithNull_ShouldThrowException()
    {
        // Arrange
        var layer = CreateTestLayer();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => layer.PreviousLayer = null!);
    }

    [Fact]
    public void SetNextLayer_WithNull_ShouldThrowException()
    {
        // Arrange
        var layer = CreateTestLayer();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => layer.NextLayer = null!);
    }
}
