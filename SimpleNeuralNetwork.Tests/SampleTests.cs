using System;
using Xunit;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class SampleTests
    {
        #region Addition Operation Tests

        [Fact]
        public void Constructor_WithAdditionOperation_SetsCorrectSampleValues()
        {
            // Arrange
            double a = 5.0;
            double b = 3.0;
            double normalizer = 1.0;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(4, sample.Xample.Length);
            Assert.Equal(a, sample.Xample[0]);
            Assert.Equal(b, sample.Xample[1]);
            Assert.Equal(0, sample.Xample[2]); // Addition operation flag
            Assert.Equal(1, sample.Xample[3]); // Addition operation flag
        }

        [Fact]
        public void Constructor_WithAdditionOperation_SetsCorrectObservedValue()
        {
            // Arrange
            double a = 7.5;
            double b = 2.5;
            double normalizer = 1.0;
            double expected = a + b;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Single(sample.Observed);
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithAdditionOperation_NegativeNumbers_CalculatesCorrectly()
        {
            // Arrange
            double a = -5.0;
            double b = 3.0;
            double normalizer = 1.0;
            double expected = a + b;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithAdditionOperation_ZeroValues_CalculatesCorrectly()
        {
            // Arrange
            double a = 0.0;
            double b = 0.0;
            double normalizer = 1.0;
            double expected = 0.0;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithAdditionOperation_LargeNumbers_CalculatesCorrectly()
        {
            // Arrange
            double a = 1000000.0;
            double b = 999999.0;
            double normalizer = 1.0;
            double expected = a + b;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithAdditionOperation_DecimalNumbers_CalculatesCorrectly()
        {
            // Arrange
            double a = 1.234567;
            double b = 2.345678;
            double normalizer = 1.0;
            double expected = a + b;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0], precision: 10);
        }

        #endregion

        #region Hypotenuse Operation Tests

        [Fact]
        public void Constructor_WithHypotenuseOperation_SetsCorrectSampleValues()
        {
            // Arrange
            double a = 3.0;
            double b = 4.0;
            double normalizer = 1.0;
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(4, sample.Xample.Length);
            Assert.Equal(a, sample.Xample[0]);
            Assert.Equal(b, sample.Xample[1]);
            Assert.Equal(1, sample.Xample[2]); // Hypotenuse operation flag
            Assert.Equal(0, sample.Xample[3]); // Hypotenuse operation flag
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_SetsCorrectObservedValue()
        {
            // Arrange
            double a = 3.0;
            double b = 4.0;
            double normalizer = 1.0;
            double expected = Math.Sqrt(a * a + b * b); // Should be 5.0
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Single(sample.Observed);
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_WithNormalizer_CalculatesCorrectly()
        {
            // Arrange
            double a = 3.0;
            double b = 4.0;
            double normalizer = 2.0;
            double normalizedA = a * normalizer;
            double normalizedB = b * normalizer;
            double expected = Math.Sqrt(normalizedA * normalizedA + normalizedB * normalizedB);
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_ZeroValues_CalculatesCorrectly()
        {
            // Arrange
            double a = 0.0;
            double b = 0.0;
            double normalizer = 1.0;
            double expected = 0.0;
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_OneZeroValue_CalculatesCorrectly()
        {
            // Arrange
            double a = 5.0;
            double b = 0.0;
            double normalizer = 1.0;
            double expected = 5.0;
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_NegativeNumbers_CalculatesCorrectly()
        {
            // Arrange
            double a = -3.0;
            double b = -4.0;
            double normalizer = 1.0;
            double expected = Math.Sqrt(a * a + b * b); // Should be 5.0 (negative squared becomes positive)
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithHypotenuseOperation_FractionalNormalizer_CalculatesCorrectly()
        {
            // Arrange
            double a = 6.0;
            double b = 8.0;
            double normalizer = 0.5;
            double normalizedA = a * normalizer; // 3.0
            double normalizedB = b * normalizer; // 4.0
            double expected = Math.Sqrt(normalizedA * normalizedA + normalizedB * normalizedB); // 5.0
            
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        #endregion

        #region Property Tests

        [Fact]
        public void Xample_Property_ReturnsCorrectArray()
        {
            // Arrange
            double a = 1.0;
            double b = 2.0;
            var sample = new Sample(a, b, Operation.add, 1.0);
            
            // Act
            var result = sample.Xample;
            
            // Assert
            Assert.NotNull(result);
            Assert.Equal(4, result.Length);
            Assert.Equal(a, result[0]);
            Assert.Equal(b, result[1]);
        }

        [Fact]
        public void Observed_Property_ReturnsCorrectArray()
        {
            // Arrange
            double a = 1.0;
            double b = 2.0;
            var sample = new Sample(a, b, Operation.add, 1.0);
            
            // Act
            var result = sample.Observed;
            
            // Assert
            Assert.NotNull(result);
            Assert.Single(result);
            Assert.Equal(a + b, result[0]);
        }

        [Fact]
        public void Xample_Property_ReturnsImmutableReference()
        {
            // Arrange
            double a = 1.0;
            double b = 2.0;
            var sample = new Sample(a, b, Operation.add, 1.0);
            
            // Act
            var result1 = sample.Xample;
            var result2 = sample.Xample;
            
            // Assert
            Assert.Same(result1, result2); // Should return the same array reference
        }

        #endregion

        #region Edge Cases and Error Handling

        [Fact]
        public void Constructor_WithVeryLargeNumbers_DoesNotOverflow()
        {
            // Arrange
            double a = double.MaxValue / 10;
            double b = double.MaxValue / 10;
            double normalizer = 1.0;
            
            // Act & Assert - Should not throw
            var sample = new Sample(a, b, Operation.add, normalizer);
            Assert.Equal(a + b, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithVerySmallNumbers_MaintainsPrecision()
        {
            // Arrange
            double a = double.Epsilon;
            double b = double.Epsilon;
            double normalizer = 1.0;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(a + b, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithInfiniteValues_HandlesCorrectly()
        {
            // Arrange
            double a = double.PositiveInfinity;
            double b = 1.0;
            double normalizer = 1.0;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(double.PositiveInfinity, sample.Observed[0]);
        }

        [Fact]
        public void Constructor_WithNaNValues_HandlesCorrectly()
        {
            // Arrange
            double a = double.NaN;
            double b = 1.0;
            double normalizer = 1.0;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.True(double.IsNaN(sample.Observed[0]));
        }

        [Theory]
        [InlineData(0.0, 0.0, 1.0)]
        [InlineData(1.0, 1.0, 1.0)]
        [InlineData(-1.0, 1.0, 1.0)]
        [InlineData(0.5, 0.5, 2.0)]
        [InlineData(10.0, 5.0, 0.1)]
        public void Constructor_WithVariousInputs_AdditionOperation_CalculatesCorrectly(double a, double b, double normalizer)
        {
            // Arrange
            double expected = a + b;
            
            // Act
            var sample = new Sample(a, b, Operation.add, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0]);
        }

        [Theory]
        [InlineData(3.0, 4.0, 1.0, 5.0)]
        [InlineData(0.0, 1.0, 1.0, 1.0)]
        [InlineData(1.0, 0.0, 1.0, 1.0)]
        [InlineData(1.0, 1.0, 1.0, 1.4142135623730951)]
        [InlineData(5.0, 12.0, 1.0, 13.0)]
        public void Constructor_WithVariousInputs_HypotenuseOperation_CalculatesCorrectly(double a, double b, double normalizer, double expected)
        {
            // Act
            var sample = new Sample(a, b, Operation.hypot, normalizer);
            
            // Assert
            Assert.Equal(expected, sample.Observed[0], precision: 10);
        }

        #endregion
    }
}
