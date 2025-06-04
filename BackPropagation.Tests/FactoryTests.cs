using System;
using Xunit;
using BackPropagation.NNLib;

namespace BackPropagation.Tests;

public class NodeFactoryTests
{
    [Fact]
    public void Create_ShouldReturnValidNode()
    {
        // Arrange
        var factory = new NodeFactory();
        var layer = new Layer(0, factory, [[1.0, 2.0]], [[0.5]], ActivationFunctions.Unit);
        double[] weights = [1.0, 2.0];
        double[] bias = [0.5];

        // Act
        var node = factory.Create(layer, 0, weights, bias, ActivationFunctions.Unit);

        // Assert
        Assert.NotNull(node);
        Assert.Equal(weights, node.Weights);
        Assert.Equal(bias, node.Bias);
        Assert.Equal(ActivationFunctions.Unit, node.ActivationFunction);
    }

    [Fact]
    public void Create_WithNullLayer_ShouldThrowException()
    {
        // Arrange
        var factory = new NodeFactory();
        double[] weights = [1.0];
        double[] bias = [0.0];

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => 
            factory.Create(null!, 0, weights, bias));
    }
}

public class LayerFactoryTests
{
    [Fact]
    public void Create_InputLayer_ShouldReturnInputLayer()
    {
        // Arrange
        var factory = new LayerFactory();
        var nodeFactory = new NodeFactory();
        double[][] weights = [];
        double[][] biases = [];

        // Act
        var layer = factory.Create(0, nodeFactory, weights, biases, 
            ActivationFunctions.Unit, LayerType.Input);

        // Assert
        Assert.NotNull(layer);
        Assert.IsType<InputLayer>(layer);
    }

    [Fact]
    public void Create_OutputLayer_ShouldReturnOutputLayer()
    {
        // Arrange
        var factory = new LayerFactory();
        var nodeFactory = new NodeFactory();
        double[][] weights = [];
        double[][] biases = [];

        // Act
        var layer = factory.Create(0, nodeFactory, weights, biases, 
            ActivationFunctions.Unit, LayerType.Output);

        // Assert
        Assert.NotNull(layer);
        Assert.IsType<OutputLayer>(layer);
    }

    [Fact]
    public void Create_HiddenLayer_ShouldReturnLayer()
    {
        // Arrange
        var factory = new LayerFactory();
        var nodeFactory = new NodeFactory();
        double[][] weights = [[1.0]];
        double[][] biases = [[0.0]];

        // Act
        var layer = factory.Create(0, nodeFactory, weights, biases, 
            ActivationFunctions.Unit, LayerType.Hidden);

        // Assert
        Assert.NotNull(layer);
        Assert.IsType<Layer>(layer);
    }
}
