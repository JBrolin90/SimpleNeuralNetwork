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

        // Unit should handle extreme values
        Assert.Equal(largePositive, ActivationFunctions.Unit(largePositive));
        Assert.Equal(largeNegative, ActivationFunctions.Unit(largeNegative));

        // SoftPlus should handle extreme values
        Assert.True(double.IsPositiveInfinity(ActivationFunctions.SoftPlus(largePositive)));
        Assert.Equal(0.0, ActivationFunctions.SoftPlus(largeNegative), 5);
    }

    // Unit Activation Function Tests
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(-1, -1)]
    [InlineData(5.5, 5.5)]
    [InlineData(-10.7, -10.7)]
    public void Unit_WithVariousInputs_ReturnsInput(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.Unit(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-1)]
    [InlineData(100)]
    [InlineData(-100)]
    public void UnitDerivative_WithAnyInput_ReturnsOne(double input)
    {
        // Act
        double result = ActivationFunctions.UnitDerivative(input);

        // Assert
        Assert.Equal(1, result);
    }

    // SoftPlus Activation Function Tests
    [Fact]
    public void SoftPlus_WithZero_ReturnsLogTwo()
    {
        // Act
        double result = ActivationFunctions.SoftPlus(0);

        // Assert
        Assert.Equal(Math.Log(2), result, 7);
    }

    [Theory]
    [InlineData(1, 1.3132616875182228)]
    [InlineData(2, 2.1269280110429727)]
    [InlineData(-1, 0.31326168751822286)]
    [InlineData(-2, 0.12692801104297249)]
    public void SoftPlus_WithVariousInputs_ReturnsExpectedValues(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.SoftPlus(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(-10)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(10)]
    public void SoftPlus_WithAnyInput_ReturnsNonNegativeValue(double input)
    {
        // Act
        double result = ActivationFunctions.SoftPlus(input);

        // Assert
        Assert.True(result >= 0, $"SoftPlus({input}) = {result} should be non-negative");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-1)]
    [InlineData(5)]
    [InlineData(-5)]
    public void SoftPlusDerivative_WithVariousInputs_EqualsSigmoid(double input)
    {
        // Act
        double softPlusDerivative = ActivationFunctions.SoftPlusDerivative(input);
        double sigmoidResult = ActivationFunctions.Sigmoid(input);

        // Assert
        Assert.Equal(sigmoidResult, softPlusDerivative, 7);
    }

    // Sigmoid Derivative Tests
    [Theory]
    [InlineData(0, 0.25)]    // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    [InlineData(1, 0.19661193324148188)] // sigmoid'(1) ≈ 0.1966
    [InlineData(2, 0.10499358540350662)] // sigmoid'(2) ≈ 0.1050
    public void SigmoidDerivative_WithSigmoidInputs_ReturnsExpectedValues(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.SigmoidDerivative(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void SigmoidDerivative_WithValidInputs_ReturnsPositiveValue(double input)
    {
        // Act
        double result = ActivationFunctions.SigmoidDerivative(input);

        // Assert
        Assert.True(result >= 0, $"SigmoidDerivative({input}) = {result} should be non-negative");
    }

    // Tanh Derivative Tests
    [Theory]
    [InlineData(0, 1)]        // tanh'(0) = 1 - tanh(0)^2 = 1 - 0^2 = 1
    [InlineData(1, 0.4199743416140261)] // tanh'(1) ≈ 0.4200
    [InlineData(2, 0.07065082485316443)] // tanh'(2) ≈ 0.0707
    [InlineData(-1, 0.4199743416140261)] // tanh'(-1) ≈ 0.4200
    public void TanhDerivative_WithTanhInputs_ReturnsExpectedValues(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.TanhDerivative(input);

        // Assert
        Assert.Equal(expected, result, 7);
    }

    [Theory]
    [InlineData(-2)]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void TanhDerivative_WithValidInputs_ReturnsBetweenZeroAndOne(double input)
    {
        // Act
        double result = ActivationFunctions.TanhDerivative(input);

        // Assert
        Assert.True(result >= 0 && result <= 1, $"TanhDerivative({input}) = {result} should be between 0 and 1");
    }

    // ReLU Derivative Tests
    [Theory]
    [InlineData(0, 0)]
    [InlineData(1, 1)]
    [InlineData(5.5, 1)]
    [InlineData(100, 1)]
    public void ReLUDerivative_WithNonNegativeInputs_ReturnsOne(double input, double expected)
    {
        // Act
        double result = ActivationFunctions.ReLUDerivative(input);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(-5.5)]
    [InlineData(-100)]
    public void ReLUDerivative_WithNegativeInputs_ReturnsZero(double input)
    {
        // Act
        double result = ActivationFunctions.ReLUDerivative(input);

        // Assert
        Assert.Equal(0, result);
    }

    // LeakyReLU Derivative Tests
    [Theory]
    [InlineData(0, 0.01, 0.01)]
    [InlineData(1, 0.01, 1)]
    [InlineData(5.5, 0.01, 1)]
    [InlineData(100, 0.01, 1)]
    public void LeakyReLUDerivative_WithNonNegativeInputs_ReturnsCorrectValue(double input, double alpha, double expected)
    {
        // Act
        double result = ActivationFunctions.LeakyReLUDerivative(input, alpha);

        // Assert
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(-1, 0.01, 0.01)]
    [InlineData(-10, 0.01, 0.01)]
    [InlineData(-2, 0.1, 0.1)]
    [InlineData(-2, 0.5, 0.5)]
    public void LeakyReLUDerivative_WithNegativeInputs_ReturnsAlpha(double input, double alpha, double expected)
    {
        // Act
        double result = ActivationFunctions.LeakyReLUDerivative(input, alpha);

        // Assert
        Assert.Equal(expected, result);
    }

    // Edge Case Tests
    [Fact]
    public void ActivationFunctions_WithNaN_HandleCorrectly()
    {
        double nanInput = double.NaN;

        // Most functions should return NaN when given NaN
        Assert.True(double.IsNaN(ActivationFunctions.Unit(nanInput)));
        Assert.True(double.IsNaN(ActivationFunctions.Sigmoid(nanInput)));
        Assert.True(double.IsNaN(ActivationFunctions.Tanh(nanInput)));
        Assert.True(double.IsNaN(ActivationFunctions.ReLU(nanInput)));
        Assert.True(double.IsNaN(ActivationFunctions.LeakyReLU(nanInput, 0.01)));
    }

    [Fact]
    public void ActivationFunctions_WithPositiveInfinity_HandleCorrectly()
    {
        double positiveInfinity = double.PositiveInfinity;

        // Test expected behaviors with positive infinity
        Assert.Equal(positiveInfinity, ActivationFunctions.Unit(positiveInfinity));
        Assert.Equal(1.0, ActivationFunctions.Sigmoid(positiveInfinity));
        Assert.Equal(1.0, ActivationFunctions.Tanh(positiveInfinity));
        Assert.Equal(positiveInfinity, ActivationFunctions.ReLU(positiveInfinity));
        Assert.Equal(positiveInfinity, ActivationFunctions.LeakyReLU(positiveInfinity, 0.01));
        Assert.Equal(positiveInfinity, ActivationFunctions.SoftPlus(positiveInfinity));
    }

    [Fact]
    public void ActivationFunctions_WithNegativeInfinity_HandleCorrectly()
    {
        double negativeInfinity = double.NegativeInfinity;

        // Test expected behaviors with negative infinity
        Assert.Equal(negativeInfinity, ActivationFunctions.Unit(negativeInfinity));
        Assert.Equal(0.0, ActivationFunctions.Sigmoid(negativeInfinity));
        Assert.Equal(-1.0, ActivationFunctions.Tanh(negativeInfinity));
        Assert.Equal(0.0, ActivationFunctions.ReLU(negativeInfinity));
        Assert.Equal(negativeInfinity * 0.01, ActivationFunctions.LeakyReLU(negativeInfinity, 0.01));
        Assert.Equal(0.0, ActivationFunctions.SoftPlus(negativeInfinity));
    }

    // Consistency Tests
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(-1)]
    [InlineData(5)]
    [InlineData(-5)]
    public void ActivationFunctions_DerivativeConsistency_SigmoidAndSoftPlus(double input)
    {
        // SoftPlus derivative should equal Sigmoid
        double softPlusDerivative = ActivationFunctions.SoftPlusDerivative(input);
        double sigmoid = ActivationFunctions.Sigmoid(input);

        // Assert
        Assert.Equal(sigmoid, softPlusDerivative, 7);
    }

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.1)]
    [InlineData(0.5)]
    public void LeakyReLU_WithDifferentAlphaValues_BehavesCorrectly(double alpha)
    {
        double negativeInput = -5.0;
        double positiveInput = 5.0;

        // Act
        double negativeResult = ActivationFunctions.LeakyReLU(negativeInput, alpha);
        double positiveResult = ActivationFunctions.LeakyReLU(positiveInput, alpha);

        // Assert
        Assert.Equal(negativeInput * alpha, negativeResult);
        Assert.Equal(positiveInput, positiveResult);
    }
}
