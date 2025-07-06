using Xunit;
using System;
using System.Linq;
using BackPropagation.NNLib;


namespace SimpleNeuralNetwork.Tests;

public class InputProcessorTests
{
    private const double Tolerance = 1e-7;

    [Theory]
    [InlineData(new double[] { 1.0, 2.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.2 }, 0.1)]  // 1.0*0.5 + 2.0*(-0.3) + 0.2 = 0.1
    [InlineData(new double[] { 0.0, 0.0 }, new double[] { 0.5, -0.3 }, new double[] { 0.1 }, 0.1)]
    [InlineData(new double[] { 2.0, -1.0, 3.0 }, new double[] { 0.2, 0.4, -0.1 }, new double[] { 0.0 }, -0.3)] // 2.0*0.2 + (-1.0)*0.4 + 3.0*(-0.1) + 0.0 = -0.3
    public void ProcessInputs_WithVariousInputs_CalculatesCorrectOutput(double[] inputs, double[] weights, double[] bias, double expectedOutput)
    {
        // Arrange
        var layer = new MockLayer();
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(expectedOutput, inputProcessor.Y, 7);
    }

    [Fact]
    public void Constructor_InitializesPropertiesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        int index = 2;
        double[] weights = { 0.1, 0.2, 0.3 };
        double[] bias = { 0.5 };

        // Act
        var inputProcessor = new InputProcessor(layer, index, weights, bias);

        // Assert
        Assert.Equal(layer, inputProcessor.Layer);
        Assert.Equal(index, inputProcessor.Index);
        Assert.Equal(weights, inputProcessor.Weights);
        Assert.Equal(bias, inputProcessor.Bias);
        Assert.Empty(inputProcessor.I);
        Assert.Equal(0.0, inputProcessor.Y);
    }

    [Fact]
    public void ProcessInputs_WithSingleInput_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 2.0 };
        double[] bias = { 1.5 };
        double[] inputs = { 3.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 3.0 * 2.0 + 1.5; // 7.5
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
        Assert.Equal(inputs, inputProcessor.I);
    }

    [Fact]
    public void ProcessInputs_WithZeroInputs_ReturnsOnlyBias()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0, 2.0, 3.0 };
        double[] bias = { 0.7 };
        double[] inputs = { 0.0, 0.0, 0.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(0.7, result, Tolerance);
        Assert.Equal(0.7, inputProcessor.Y, Tolerance);
        Assert.Equal(inputs, inputProcessor.I);
    }

    [Fact]
    public void ProcessInputs_WithNegativeWeights_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { -1.0, -2.0 };
        double[] bias = { 0.0 };
        double[] inputs = { 1.0, 1.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 1.0 * (-1.0) + 1.0 * (-2.0) + 0.0; // -3.0
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_WithNegativeBias_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0, 1.0 };
        double[] bias = { -2.5 };
        double[] inputs = { 1.0, 1.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 1.0 * 1.0 + 1.0 * 1.0 + (-2.5); // -0.5
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_WithLargeNumbers_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1000.0, 2000.0 };
        double[] bias = { 500.0 };
        double[] inputs = { 0.001, 0.002 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 0.001 * 1000.0 + 0.002 * 2000.0 + 500.0; // 505.0
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_WithVerySmallNumbers_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 0.0001, 0.0002 };
        double[] bias = { 0.0003 };
        double[] inputs = { 10.0, 20.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 10.0 * 0.0001 + 20.0 * 0.0002 + 0.0003; // 0.0063
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_MultipleCalls_UpdatesStateCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0, 2.0 };
        double[] bias = { 0.5 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act & Assert - First call
        double[] inputs1 = { 1.0, 2.0 };
        double result1 = inputProcessor.ProcessInputs(inputs1);
        double expected1 = 1.0 * 1.0 + 2.0 * 2.0 + 0.5; // 5.5
        Assert.Equal(expected1, result1, Tolerance);
        Assert.Equal(expected1, inputProcessor.Y, Tolerance);
        Assert.Equal(inputs1, inputProcessor.I);

        // Act & Assert - Second call
        double[] inputs2 = { 3.0, 4.0 };
        double result2 = inputProcessor.ProcessInputs(inputs2);
        double expected2 = 3.0 * 1.0 + 4.0 * 2.0 + 0.5; // 11.5
        Assert.Equal(expected2, result2, Tolerance);
        Assert.Equal(expected2, inputProcessor.Y, Tolerance);
        Assert.Equal(inputs2, inputProcessor.I);
    }

    [Fact]
    public void ProcessInputs_StoresInputsCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 0.5, 0.3 };
        double[] bias = { 0.1 };
        double[] inputs = { 2.0, 3.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(inputs, inputProcessor.I);
        Assert.Same(inputs, inputProcessor.I); // Should be the same reference
    }

    [Theory]
    [InlineData(new double[] { 1.0 }, new double[] { 0.0 }, new double[] { 0.0 }, 0.0)]
    [InlineData(new double[] { 0.0 }, new double[] { 1.0 }, new double[] { 0.0 }, 0.0)]
    [InlineData(new double[] { 2.0 }, new double[] { 0.0 }, new double[] { 3.0 }, 3.0)]
    [InlineData(new double[] { 0.0 }, new double[] { 0.0 }, new double[] { 1.0 }, 1.0)]
    public void ProcessInputs_EdgeCases_CalculatesCorrectly(double[] inputs, double[] weights, double[] bias, double expected)
    {
        // Arrange
        var layer = new MockLayer();
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_WithFractionalNumbers_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 0.25, 0.75 };
        double[] bias = { 0.125 };
        double[] inputs = { 0.5, 0.25 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 0.5 * 0.25 + 0.25 * 0.75 + 0.125; // 0.4375
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void ProcessInputs_WithMixedSignInputs_CalculatesCorrectly()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0, -1.0, 2.0 };
        double[] bias = { 0.0 };
        double[] inputs = { 5.0, -3.0, 2.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 5.0 * 1.0 + (-3.0) * (-1.0) + 2.0 * 2.0 + 0.0; // 12.0
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    [Fact]
    public void InputProcessor_Properties_CanBeSetAndRetrieved()
    {
        // Arrange
        var layer1 = new MockLayer();
        var layer2 = new MockLayer();
        double[] weights1 = { 1.0, 2.0 };
        double[] weights2 = { 3.0, 4.0 };
        double[] bias1 = { 0.5 };
        double[] bias2 = { 1.0 };
        var inputProcessor = new InputProcessor(layer1, 0, weights1, bias1);

        // Act & Assert - Initial values
        Assert.Equal(layer1, inputProcessor.Layer);
        Assert.Equal(0, inputProcessor.Index);
        Assert.Equal(weights1, inputProcessor.Weights);
        Assert.Equal(bias1, inputProcessor.Bias);

        // Act & Assert - Modified values
        inputProcessor.Layer = layer2;
        inputProcessor.Index = 5;
        inputProcessor.Weights = weights2;
        inputProcessor.Bias = bias2;

        Assert.Equal(layer2, inputProcessor.Layer);
        Assert.Equal(5, inputProcessor.Index);
        Assert.Equal(weights2, inputProcessor.Weights);
        Assert.Equal(bias2, inputProcessor.Bias);
    }

    #region InputProcessorFactory Tests

    [Fact]
    public void InputProcessorFactory_Build_CreatesInputProcessorCorrectly()
    {
        // Arrange
        var factory = new InputProcessorFactory();
        var layer = new MockLayer();
        int index = 3;
        double[] weights = { 0.1, 0.2, 0.3 };
        double[] bias = { 0.5 };

        // Act
        var inputProcessor = factory.Build(layer, index, weights, bias);

        // Assert
        Assert.NotNull(inputProcessor);
        Assert.IsType<InputProcessor>(inputProcessor);
        Assert.Equal(layer, inputProcessor.Layer);
        Assert.Equal(index, inputProcessor.Index);
        Assert.Equal(weights, inputProcessor.Weights);
        Assert.Equal(bias, inputProcessor.Bias);
    }

    [Fact]
    public void InputProcessorFactory_Build_WithNullLayer_CreatesInputProcessor()
    {
        // Arrange
        var factory = new InputProcessorFactory();
        ILayer? layer = null;
        int index = 1;
        double[] weights = { 1.0 };
        double[] bias = { 0.0 };

        // Act
        var inputProcessor = factory.Build(layer!, index, weights, bias);

        // Assert
        Assert.NotNull(inputProcessor);
        Assert.Null(inputProcessor.Layer);
        Assert.Equal(index, inputProcessor.Index);
        Assert.Equal(weights, inputProcessor.Weights);
        Assert.Equal(bias, inputProcessor.Bias);
    }

    [Fact]
    public void InputProcessorFactory_Build_WithEmptyWeights_CreatesInputProcessor()
    {
        // Arrange
        var factory = new InputProcessorFactory();
        var layer = new MockLayer();
        int index = 0;
        double[] weights = Array.Empty<double>();
        double[] bias = { 1.0 };

        // Act
        var inputProcessor = factory.Build(layer, index, weights, bias);

        // Assert
        Assert.NotNull(inputProcessor);
        Assert.Equal(layer, inputProcessor.Layer);
        Assert.Equal(index, inputProcessor.Index);
        Assert.Equal(weights, inputProcessor.Weights);
        Assert.Empty(inputProcessor.Weights);
        Assert.Equal(bias, inputProcessor.Bias);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void ProcessInputs_WithEmptyInputs_ReturnsOnlyBias()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = Array.Empty<double>();
        double[] bias = { 2.5 };
        double[] inputs = Array.Empty<double>();
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(2.5, result, Tolerance);
        Assert.Equal(2.5, inputProcessor.Y, Tolerance);
        Assert.Equal(inputs, inputProcessor.I);
    }

    [Fact]
    public void ProcessInputs_WithMismatchedInputWeightsLength_ProcessesAvailableInputs()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0, 2.0, 3.0 };
        double[] bias = { 0.0 };
        double[] inputs = { 5.0, 10.0 }; // Only 2 inputs for 3 weights
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        // Should only process the first 2 inputs: 5.0*1.0 + 10.0*2.0 = 25.0
        Assert.Equal(25.0, result, Tolerance);
        Assert.Equal(25.0, inputProcessor.Y, Tolerance);
    }

    [Theory]
    [InlineData(double.MaxValue)]
    [InlineData(double.MinValue)]
    [InlineData(double.Epsilon)]
    [InlineData(-double.Epsilon)]
    public void ProcessInputs_WithExtremeValues_HandlesCorrectly(double extremeValue)
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0 };
        double[] bias = { 0.0 };
        double[] inputs = { extremeValue };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        Assert.Equal(extremeValue, result);
        Assert.Equal(extremeValue, inputProcessor.Y);
    }

    #endregion

    #region Interface Implementation Tests

    [Fact]
    public void InputProcessor_ImplementsIInputProcessor()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0 };
        double[] bias = { 0.0 };
        
        // Act
        IInputProcessor inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Assert
        Assert.IsAssignableFrom<IInputProcessor>(inputProcessor);
        Assert.Equal(layer, inputProcessor.Layer);
        Assert.Equal(0, inputProcessor.Index);
        Assert.Equal(weights, inputProcessor.Weights);
        Assert.Equal(bias, inputProcessor.Bias);
        Assert.Empty(inputProcessor.I);
        Assert.Equal(0.0, inputProcessor.Y);
    }

    [Fact]
    public void InputProcessor_InterfaceProperties_CanBeModified()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.0 };
        double[] bias = { 0.0 };
        IInputProcessor inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        inputProcessor.Y = 5.5;
        inputProcessor.I = new double[] { 1.0, 2.0, 3.0 };
        inputProcessor.Index = 10;

        // Assert
        Assert.Equal(5.5, inputProcessor.Y);
        Assert.Equal(new double[] { 1.0, 2.0, 3.0 }, inputProcessor.I);
        Assert.Equal(10, inputProcessor.Index);
    }

    [Fact]
    public void InputProcessor_ProcessInputs_ReturnsCorrectValueThroughInterface()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 2.0, 3.0 };
        double[] bias = { 1.0 };
        double[] inputs = { 4.0, 5.0 };
        IInputProcessor inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 4.0 * 2.0 + 5.0 * 3.0 + 1.0; // 24.0
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    #endregion

    #region Performance and State Tests

    [Fact]
    public void ProcessInputs_MultipleCallsWithSameInputs_ConsistentResults()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = { 1.5, 2.5 };
        double[] bias = { 0.5 };
        double[] inputs = { 2.0, 3.0 };
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result1 = inputProcessor.ProcessInputs(inputs);
        double result2 = inputProcessor.ProcessInputs(inputs);
        double result3 = inputProcessor.ProcessInputs(inputs);

        // Assert
        double expected = 2.0 * 1.5 + 3.0 * 2.5 + 0.5; // 11.5
        Assert.Equal(expected, result1, Tolerance);
        Assert.Equal(expected, result2, Tolerance);
        Assert.Equal(expected, result3, Tolerance);
        Assert.Equal(result1, result2);
        Assert.Equal(result2, result3);
    }

    [Fact]
    public void ProcessInputs_WithLargeInputArray_ProcessesEfficiently()
    {
        // Arrange
        var layer = new MockLayer();
        double[] weights = Enumerable.Range(0, 1000).Select(i => (double)i / 1000.0).ToArray();
        double[] bias = { 0.5 };
        double[] inputs = Enumerable.Range(0, 1000).Select(i => (double)i).ToArray();
        var inputProcessor = new InputProcessor(layer, 0, weights, bias);

        // Act
        double result = inputProcessor.ProcessInputs(inputs);

        // Assert
        // Manual calculation: sum of i * (i/1000) for i from 0 to 999, plus 0.5
        double expected = 0.0;
        for (int i = 0; i < 1000; i++)
        {
            expected += i * (i / 1000.0);
        }
        expected += 0.5;
        
        Assert.Equal(expected, result, Tolerance);
        Assert.Equal(expected, inputProcessor.Y, Tolerance);
    }

    #endregion
}
