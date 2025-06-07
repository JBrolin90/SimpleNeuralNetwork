using Xunit;
using System;
using BackPropagation.NNLib;


namespace SimpleNeuralNetwork.Tests;

public class ActivationFunctionsTests
{
    private const double Tolerance = 1e-7;

    [Fact]
    public void Sigmoid_WithZero_ReturnsHalf()
    {
        // Act
        double result = ActivationFunctions.Sigmoid(0);

        // Assert
        Assert.Equal(0.5, result, 7);
    }

    [Theory]
    [InlineData(1, 0.7310585786300049)]
    [InlineData(2, 0.8807970779778823)]
    [InlineData(-1, 0.2689414213699951)]
    [InlineData(-2, 0.11920292202211755)]
    public void Sigmoid_WithVariousInputs_ReturnsExpectedValues(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.Sigmoid(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(-100)]
    [InlineData(-10)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    public void Sigmoid_WithAnyInput_OutputBetweenZeroAndOne(double input)
    {
        // Act
        double result = ActivationFunctions.Sigmoid(input);

        // Assert
        Assert.True(result >= 0 && result <= 1, $"Sigmoid({input}) = {result} should be between 0 and 1 (inclusive)");
    }

    [Fact]
    public void Tanh_WithZero_ReturnsZero()
    {
        // Act
        double result = ActivationFunctions.Tanh(0);

        // Assert
        Assert.Equal(0, result, 7);
    }

    [Theory]
    [InlineData(1, 0.7615941559557649)]
    [InlineData(2, 0.9640275800758169)]
    [InlineData(-1, -0.7615941559557649)]
    [InlineData(-2, -0.9640275800758169)]
    public void Tanh_WithVariousInputs_ReturnsExpectedValues(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.Tanh(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(-100)]
    [InlineData(-10)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(10)]
    [InlineData(100)]
    public void Tanh_WithAnyInput_OutputBetweenMinusOneAndOne(double input)
    {
        // Act
        double result = ActivationFunctions.Tanh(input);

        // Assert
        Assert.True(result >= -1 && result <= 1);
    }

    [Theory]
    [InlineData(0.5)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public void Tanh_IsOddFunction_SymmetricAroundZero(double input)
    {
        // Act
        double positive = ActivationFunctions.Tanh(input);
        double negative = ActivationFunctions.Tanh(-input);

        // Assert
        Assert.Equal(-positive, negative, 7);
    }

    [Fact]
    public void ReLU_WithZero_ReturnsZero()
    {
        // Act
        double result = ActivationFunctions.ReLU(0);

        // Assert
        Assert.Equal(0, result);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5.5, 5.5)]
    [InlineData(100, 100)]
    public void ReLU_WithPositiveInputs_ReturnsInput(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.ReLU(input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-5.5)]
    [InlineData(-100)]
    public void ReLU_WithNegativeInputs_ReturnsZero(double input)
    {
        // Act
        double result = ActivationFunctions.ReLU(input);

        // Assert
        Assert.Equal(0, result);
    }

    [Fact]
    public void LeakyReLU_WithZero_ReturnsZero()
    {
        // Act
        double result = ActivationFunctions.LeakyReLU(0, 0.01);

        // Assert
        Assert.Equal(0, result);
    }

    [Theory]
    [InlineData(1, 0.01, 1)]
    [InlineData(5.5, 0.01, 5.5)]
    public void LeakyReLU_WithPositiveInputs_ReturnsInput(double input, double alpha, double expected)
    {
        // Act
        double result = ActivationFunctions.LeakyReLU(input, alpha);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(-1, 0.01, -0.01)]
    [InlineData(-10, 0.01, -0.1)]
    [InlineData(-2, 0.1, -0.2)]
    [InlineData(-2, 0.5, -1.0)]
    public void LeakyReLU_WithNegativeInputs_ReturnsScaledInput(double input, double alpha, double expected)
    {
        // Act
        double result = ActivationFunctions.LeakyReLU(input, alpha);

        // Assert
        Assert.Equal(expected, result);
    }

    [Fact]
    public void ActivationFunctions_WithExtremeValues_HandleCorrectly()
    {
        double largePositive = 1000;
        double largeNegative = -1000;

        // Sigmoid should approach 0 and 1
        Assert.Equal(1.0, ActivationFunctions.Sigmoid(largePositive), 5);
        Assert.Equal(0.0, ActivationFunctions.Sigmoid(largeNegative), 5);

        // Tanh should approach 1 and -1
        Assert.Equal(1.0, ActivationFunctions.Tanh(largePositive), 5);
        Assert.Equal(-1.0, ActivationFunctions.Tanh(largeNegative), 5);

        // ReLU should handle extreme values
        Assert.Equal(largePositive, ActivationFunctions.ReLU(largePositive));
        Assert.Equal(0, ActivationFunctions.ReLU(largeNegative));

        // Leaky ReLU should handle extreme values
        double alpha = 0.01;
        Assert.Equal(largePositive, ActivationFunctions.LeakyReLU(largePositive, alpha));
        Assert.Equal(largeNegative * alpha, ActivationFunctions.LeakyReLU(largeNegative, alpha));
    }
}
