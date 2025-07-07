using System;
using Xunit;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class LossFunctionsTests
    {
        #region SquaredError Tests

        [Fact]
        public void SquaredError_WithEqualArrays_ReturnsZeroErrors()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0, 3.0 };
            double[] observed = { 1.0, 2.0, 3.0 };
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            Assert.All(result, error => Assert.Equal(0.0, error));
        }

        [Fact]
        public void SquaredError_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 2.0, 4.0, 6.0 };
            double[] observed = { 1.0, 2.0, 3.0 };
            double[] expected = { 1.0, 4.0, 9.0 }; // (2-1)² = 1, (4-2)² = 4, (6-3)² = 9
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SquaredError_WithNegativeValues_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { -1.0, 0.0, 1.0 };
            double[] observed = { 1.0, 2.0, -1.0 };
            double[] expected = { 4.0, 4.0, 4.0 }; // (-1-1)² = 4, (0-2)² = 4, (1-(-1))² = 4
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SquaredError_WithSingleElement_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 5.0 };
            double[] observed = { 2.0 };
            double expected = 9.0; // (5-2)² = 9
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, observed);
            
            // Assert
            Assert.Single(result);
            Assert.Equal(expected, result[0]);
        }

        [Fact]
        public void SquaredError_WithEmptyArrays_ReturnsEmptyArray()
        {
            // Arrange
            double[] predicted = { };
            double[] observed = { };
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, observed);
            
            // Assert
            Assert.Empty(result);
        }

        [Fact]
        public void SquaredError_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0, 3.0 };
            double[] observed = { 1.0, 2.0 };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SquaredError(predicted, observed));
        }

        #endregion

        #region SquaredErrorDerivative Tests

        [Fact]
        public void SquaredErrorDerivative_WithEqualArrays_ReturnsZeroDerivatives()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0, 3.0 };
            double[] observed = { 1.0, 2.0, 3.0 };
            
            // Act
            double[] result = LossFunctions.SquaredErrorDerivative(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            Assert.All(result, derivative => Assert.Equal(0.0, derivative));
        }

        [Fact]
        public void SquaredErrorDerivative_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 2.0, 4.0, 6.0 };
            double[] observed = { 1.0, 2.0, 3.0 };
            double[] expected = { 2.0, 4.0, 6.0 }; // 2*(2-1) = 2, 2*(4-2) = 4, 2*(6-3) = 6
            
            // Act
            double[] result = LossFunctions.SquaredErrorDerivative(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SquaredErrorDerivative_WithNegativeValues_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { -1.0, 0.0, 1.0 };
            double[] observed = { 1.0, 2.0, -1.0 };
            double[] expected = { -4.0, -4.0, 4.0 }; // 2*(-1-1) = -4, 2*(0-2) = -4, 2*(1-(-1)) = 4
            
            // Act
            double[] result = LossFunctions.SquaredErrorDerivative(predicted, observed);
            
            // Assert
            Assert.Equal(3, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SquaredErrorDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0, 3.0 };
            double[] observed = { 1.0, 2.0 };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SquaredErrorDerivative(predicted, observed));
        }

        #endregion

        #region SumSquaredError Tests

        [Fact]
        public void SumSquaredError_WithEqualArrays_ReturnsZeroSum()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            
            // Act
            double[] result = LossFunctions.SumSquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(2, result.Length);
            Assert.All(result, sum => Assert.Equal(0.0, sum));
        }

        [Fact]
        public void SumSquaredError_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 2.0, 4.0 }, new double[] { 6.0, 8.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            // Expected: First output: (2-1)² + (6-3)² = 1 + 9 = 10
            //          Second output: (4-2)² + (8-4)² = 4 + 16 = 20
            double[] expected = { 10.0, 20.0 };
            
            // Act
            double[] result = LossFunctions.SumSquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumSquaredError_WithSinglePrediction_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 5.0, 10.0 } };
            double[][] observed = { new double[] { 2.0, 6.0 } };
            double[] expected = { 9.0, 16.0 }; // (5-2)² = 9, (10-6)² = 16
            
            // Act
            double[] result = LossFunctions.SumSquaredError(predicted, observed);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumSquaredError_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 } };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SumSquaredError(predicted, observed));
        }

        #endregion

        #region SumSquaredErrorDerivative Tests

        [Fact]
        public void SumSquaredErrorDerivative_WithEqualArrays_ReturnsZeroDerivatives()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            
            // Act
            double[] result = LossFunctions.SumSquaredErrorDerivative(predicted, observed);
            
            // Assert
            Assert.Equal(2, result.Length);
            Assert.All(result, derivative => Assert.Equal(0.0, derivative));
        }

        [Fact]
        public void SumSquaredErrorDerivative_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 2.0, 4.0 }, new double[] { 6.0, 8.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            // Expected: First output: 2*(2-1) + 2*(6-3) = 2 + 6 = 8
            //          Second output: 2*(4-2) + 2*(8-4) = 4 + 8 = 12
            double[] expected = { 8.0, 12.0 };
            
            // Act
            double[] result = LossFunctions.SumSquaredErrorDerivative(predicted, observed);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumSquaredErrorDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] observed = { new double[] { 1.0, 2.0 } };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SumSquaredErrorDerivative(predicted, observed));
        }

        #endregion

        #region SumMeanSquaredError Tests

        [Fact]
        public void SumMeanSquaredError_WithEqualArrays_ReturnsZeroMean()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            
            // Act
            double[] result = LossFunctions.SumMeanSquaredError(predicted, actual);
            
            // Assert
            Assert.Equal(2, result.Length);
            Assert.All(result, mean => Assert.Equal(0.0, mean));
        }

        [Fact]
        public void SumMeanSquaredError_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 2.0, 4.0 }, new double[] { 6.0, 8.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            // Expected: First output: ((2-1)² + (6-3)²) / 2 = (1 + 9) / 2 = 5
            //          Second output: ((4-2)² + (8-4)²) / 2 = (4 + 16) / 2 = 10
            double[] expected = { 5.0, 10.0 };
            
            // Act
            double[] result = LossFunctions.SumMeanSquaredError(predicted, actual);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumMeanSquaredError_WithSinglePrediction_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 5.0, 10.0 } };
            double[][] actual = { new double[] { 2.0, 6.0 } };
            double[] expected = { 9.0, 16.0 }; // (5-2)² / 1 = 9, (10-6)² / 1 = 16
            
            // Act
            double[] result = LossFunctions.SumMeanSquaredError(predicted, actual);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumMeanSquaredError_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 } };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SumMeanSquaredError(predicted, actual));
        }

        #endregion

        #region SumMeanSquaredErrorDerivative Tests

        [Fact]
        public void SumMeanSquaredErrorDerivative_WithEqualArrays_ReturnsZeroDerivatives()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            
            // Act
            double[] result = LossFunctions.SumMeanSquaredErrorDerivative(predicted, actual);
            
            // Assert
            Assert.Equal(2, result.Length);
            Assert.All(result, derivative => Assert.Equal(0.0, derivative));
        }

        [Fact]
        public void SumMeanSquaredErrorDerivative_WithDifferentArrays_CalculatesCorrectly()
        {
            // Arrange
            double[][] predicted = { new double[] { 2.0, 4.0 }, new double[] { 6.0, 8.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            // Expected: First output: (2*(2-1) + 2*(6-3)) / 2 = (2 + 6) / 2 = 4
            //          Second output: (2*(4-2) + 2*(8-4)) / 2 = (4 + 8) / 2 = 6
            double[] expected = { 4.0, 6.0 };
            
            // Act
            double[] result = LossFunctions.SumMeanSquaredErrorDerivative(predicted, actual);
            
            // Assert
            Assert.Equal(2, result.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], result[i]);
            }
        }

        [Fact]
        public void SumMeanSquaredErrorDerivative_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[][] predicted = { new double[] { 1.0, 2.0 }, new double[] { 3.0, 4.0 } };
            double[][] actual = { new double[] { 1.0, 2.0 } };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.SumMeanSquaredErrorDerivative(predicted, actual));
        }

        #endregion

        #region CrossEntropyLoss Tests

        [Fact]
        public void CrossEntropyLoss_WithPerfectPrediction_ReturnsZeroLoss()
        {
            // Arrange
            double[] predicted = { 1.0, 0.0, 0.0 };
            double[] actual = { 1.0, 0.0, 0.0 };
            
            // Act
            double result = LossFunctions.CrossEntropyLoss(predicted, actual);
            
            // Assert
            Assert.Equal(0.0, result, precision: 10);
        }

        [Fact]
        public void CrossEntropyLoss_WithTypicalPrediction_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 0.8, 0.1, 0.1 };
            double[] actual = { 1.0, 0.0, 0.0 };
            // Expected: -1 * log(0.8) / 3 ≈ 0.074
            
            // Act
            double result = LossFunctions.CrossEntropyLoss(predicted, actual);
            
            // Assert
            Assert.True(result > 0);
            Assert.Equal(-Math.Log(0.8) / 3, result, precision: 10);
        }

        [Fact]
        public void CrossEntropyLoss_WithZeroPrediction_HandlesLogOfZero()
        {
            // Arrange
            double[] predicted = { 0.0, 0.5, 0.5 };
            double[] actual = { 1.0, 0.0, 0.0 };
            // Should clamp 0.0 to 1e-15 to avoid log(0)
            
            // Act
            double result = LossFunctions.CrossEntropyLoss(predicted, actual);
            
            // Assert
            Assert.True(result > 0);
            Assert.False(double.IsInfinity(result));
            Assert.False(double.IsNaN(result));
        }

        [Fact]
        public void CrossEntropyLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[] predicted = { 0.8, 0.1, 0.1 };
            double[] actual = { 1.0, 0.0 };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.CrossEntropyLoss(predicted, actual));
        }

        #endregion

        #region HingeLoss Tests

        [Fact]
        public void HingeLoss_WithPerfectPrediction_ReturnsZeroLoss()
        {
            // Arrange
            double[] predicted = { 2.0, -2.0 };
            double[] actual = { 1.0, -1.0 };
            // Margin: 1*2 = 2, -1*(-2) = 2
            // Max(0, 1-2) = 0, Max(0, 1-2) = 0
            
            // Act
            double result = LossFunctions.HingeLoss(predicted, actual);
            
            // Assert
            Assert.Equal(0.0, result);
        }

        [Fact]
        public void HingeLoss_WithIncorrectPrediction_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 0.5, 0.5 };
            double[] actual = { 1.0, -1.0 };
            // Margin: 1*0.5 = 0.5, -1*0.5 = -0.5
            // Max(0, 1-0.5) = 0.5, Max(0, 1-(-0.5)) = 1.5
            // Average: (0.5 + 1.5) / 2 = 1.0
            
            // Act
            double result = LossFunctions.HingeLoss(predicted, actual);
            
            // Assert
            Assert.Equal(1.0, result);
        }

        [Fact]
        public void HingeLoss_WithNegativeMargin_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { -1.0, 1.0 };
            double[] actual = { 1.0, -1.0 };
            // Margin: 1*(-1) = -1, -1*1 = -1
            // Max(0, 1-(-1)) = 2, Max(0, 1-(-1)) = 2
            // Average: (2 + 2) / 2 = 2.0
            
            // Act
            double result = LossFunctions.HingeLoss(predicted, actual);
            
            // Assert
            Assert.Equal(2.0, result);
        }

        [Fact]
        public void HingeLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[] predicted = { 0.5, 0.5 };
            double[] actual = { 1.0 };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.HingeLoss(predicted, actual));
        }

        #endregion

        #region HuberLoss Tests

        [Fact]
        public void HuberLoss_WithPerfectPrediction_ReturnsZeroLoss()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0, 3.0 };
            double[] actual = { 1.0, 2.0, 3.0 };
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual);
            
            // Assert
            Assert.Equal(0.0, result);
        }

        [Fact]
        public void HuberLoss_WithSmallDifferences_UsesSquaredError()
        {
            // Arrange
            double[] predicted = { 1.5, 2.5 };
            double[] actual = { 1.0, 2.0 };
            double delta = 1.0;
            // |diff| = 0.5 <= delta, so use 0.5 * diff²
            // Loss per element: 0.5 * 0.5² = 0.125
            // Average: (0.125 + 0.125) / 2 = 0.125
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual, delta);
            
            // Assert
            Assert.Equal(0.125, result);
        }

        [Fact]
        public void HuberLoss_WithLargeDifferences_UsesLinearError()
        {
            // Arrange
            double[] predicted = { 3.0, -1.0 };
            double[] actual = { 1.0, 1.0 };
            double delta = 1.0;
            // |diff| = 2.0 > delta, so use delta * (|diff| - 0.5 * delta)
            // Loss per element: 1.0 * (2.0 - 0.5 * 1.0) = 1.0 * 1.5 = 1.5
            // Average: (1.5 + 1.5) / 2 = 1.5
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual, delta);
            
            // Assert
            Assert.Equal(1.5, result);
        }

        [Fact]
        public void HuberLoss_WithMixedDifferences_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 1.5, 3.0 };
            double[] actual = { 1.0, 1.0 };
            double delta = 1.0;
            // First: |diff| = 0.5 <= delta, so 0.5 * 0.5² = 0.125
            // Second: |diff| = 2.0 > delta, so 1.0 * (2.0 - 0.5) = 1.5
            // Average: (0.125 + 1.5) / 2 = 0.8125
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual, delta);
            
            // Assert
            Assert.Equal(0.8125, result);
        }

        [Fact]
        public void HuberLoss_WithCustomDelta_CalculatesCorrectly()
        {
            // Arrange
            double[] predicted = { 3.0 };
            double[] actual = { 1.0 };
            double delta = 0.5;
            // |diff| = 2.0 > delta, so delta * (|diff| - 0.5 * delta)
            // Loss: 0.5 * (2.0 - 0.5 * 0.5) = 0.5 * 1.75 = 0.875
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual, delta);
            
            // Assert
            Assert.Equal(0.875, result);
        }

        [Fact]
        public void HuberLoss_WithDefaultDelta_UsesDefaultValue()
        {
            // Arrange
            double[] predicted = { 2.0 };
            double[] actual = { 1.0 };
            // Default delta = 1.0
            // |diff| = 1.0 = delta, so use squared error: 0.5 * 1.0² = 0.5
            
            // Act
            double result = LossFunctions.HuberLoss(predicted, actual);
            
            // Assert
            Assert.Equal(0.5, result);
        }

        [Fact]
        public void HuberLoss_WithDifferentLengths_ThrowsArgumentException()
        {
            // Arrange
            double[] predicted = { 1.0, 2.0 };
            double[] actual = { 1.0 };
            
            // Act & Assert
            Assert.Throws<ArgumentException>(() => LossFunctions.HuberLoss(predicted, actual));
        }

        #endregion

        #region Edge Cases and Error Handling

        [Fact]
        public void SquaredError_WithVeryLargeNumbers_HandlesCorrectly()
        {
            // Arrange
            double[] predicted = { 1e100 };
            double[] actual = { 0.0 };
            
            // Act
            double[] result = LossFunctions.SquaredError(predicted, actual);
            
            // Assert
            Assert.Single(result);
            Assert.True(result[0] > 0);
            // With very large numbers, result might be infinity due to overflow
            Assert.True(double.IsFinite(result[0]) || double.IsInfinity(result[0]));
        }

        [Fact]
        public void CrossEntropyLoss_WithVerySmallPredictions_HandlesCorrectly()
        {
            // Arrange
            double[] predicted = { 1e-20, 1e-20 };
            double[] actual = { 1.0, 0.0 };
            
            // Act
            double result = LossFunctions.CrossEntropyLoss(predicted, actual);
            
            // Assert
            Assert.True(result > 0);
            Assert.False(double.IsInfinity(result));
            Assert.False(double.IsNaN(result));
        }

        [Theory]
        [InlineData(0.0, 0.0)]
        [InlineData(1.0, 1.0)]
        [InlineData(-1.0, 1.0)]
        [InlineData(0.5, 0.25)]
        [InlineData(-0.5, 0.25)]
        public void SquaredError_WithVariousInputs_CalculatesCorrectly(double predicted, double expected)
        {
            // Act
            double[] result = LossFunctions.SquaredError(new double[] { predicted }, new double[] { 0.0 });
            
            // Assert
            Assert.Equal(expected, result[0]);
        }

        #endregion
    }
}
