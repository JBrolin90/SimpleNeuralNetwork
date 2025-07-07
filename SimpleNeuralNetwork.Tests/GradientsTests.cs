using Xunit;
using System;
using BackPropagation.NNLib;

namespace SimpleNeuralNetwork.Tests
{
    public class GradientsTests
    {
        private const double Tolerance = 1e-7;

        #region Constructor Tests

        [Fact]
        public void Gradients_Constructor_InitializesPropertiesCorrectly()
        {
            // Arrange
            int weightCount = 3;
            
            // Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            Assert.NotNull(gradients.WeightGradient);
            Assert.Equal(weightCount, gradients.WeightGradient.Length);
            Assert.Equal(0.0, gradients.BiasGradient);
        }

        [Fact]
        public void Gradients_Constructor_InitializesWeightGradientArrayToZero()
        {
            // Arrange
            int weightCount = 5;
            
            // Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            for (int i = 0; i < weightCount; i++)
            {
                Assert.Equal(0.0, gradients.WeightGradient[i]);
            }
        }

        [Fact]
        public void Gradients_Constructor_WithZeroWeightCount_CreatesEmptyArray()
        {
            // Arrange
            int weightCount = 0;
            
            // Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            Assert.NotNull(gradients.WeightGradient);
            Assert.Empty(gradients.WeightGradient);
            Assert.Equal(0.0, gradients.BiasGradient);
        }

        [Fact]
        public void Gradients_Constructor_WithNegativeWeightCount_ThrowsException()
        {
            // Arrange
            int weightCount = -1;
            
            // Act & Assert
            Assert.Throws<OverflowException>(() => new Gradients(weightCount));
        }

        [Fact]
        public void Gradients_Constructor_WithLargeWeightCount_CreatesCorrectSizeArray()
        {
            // Arrange
            int weightCount = 1000;
            
            // Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            Assert.Equal(weightCount, gradients.WeightGradient.Length);
            Assert.All(gradients.WeightGradient, gradient => Assert.Equal(0.0, gradient));
        }

        #endregion

        #region Property Tests

        [Fact]
        public void WeightGradient_CanBeModified()
        {
            // Arrange
            var gradients = new Gradients(3);
            double[] expectedValues = { 0.5, -0.3, 1.2 };
            
            // Act
            for (int i = 0; i < expectedValues.Length; i++)
            {
                gradients.WeightGradient[i] = expectedValues[i];
            }
            
            // Assert
            for (int i = 0; i < expectedValues.Length; i++)
            {
                Assert.Equal(expectedValues[i], gradients.WeightGradient[i], Tolerance);
            }
        }

        [Fact]
        public void BiasGradient_CanBeModified()
        {
            // Arrange
            var gradients = new Gradients(2);
            double expectedBias = 0.75;
            
            // Act
            gradients.BiasGradient = expectedBias;
            
            // Assert
            Assert.Equal(expectedBias, gradients.BiasGradient, Tolerance);
        }

        [Fact]
        public void BiasGradient_DefaultValueIsZero()
        {
            // Arrange & Act
            var gradients = new Gradients(1);
            
            // Assert
            Assert.Equal(0.0, gradients.BiasGradient);
        }

        [Fact]
        public void WeightGradient_IsNotNull()
        {
            // Arrange & Act
            var gradients = new Gradients(1);
            
            // Assert
            Assert.NotNull(gradients.WeightGradient);
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Gradients_WithSingleWeight_WorksCorrectly()
        {
            // Arrange
            int weightCount = 1;
            
            // Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            Assert.Single(gradients.WeightGradient);
            Assert.Equal(0.0, gradients.WeightGradient[0]);
        }

        [Fact]
        public void Gradients_WeightGradientArray_IsIndependent()
        {
            // Arrange
            var gradients1 = new Gradients(2);
            var gradients2 = new Gradients(2);
            
            // Act
            gradients1.WeightGradient[0] = 1.0;
            gradients2.WeightGradient[0] = 2.0;
            
            // Assert
            Assert.Equal(1.0, gradients1.WeightGradient[0]);
            Assert.Equal(2.0, gradients2.WeightGradient[0]);
            Assert.NotSame(gradients1.WeightGradient, gradients2.WeightGradient);
        }

        [Fact]
        public void Gradients_CanHandleExtremeValues()
        {
            // Arrange
            var gradients = new Gradients(2);
            double largeValue = double.MaxValue;
            double smallValue = double.MinValue;
            
            // Act
            gradients.WeightGradient[0] = largeValue;
            gradients.WeightGradient[1] = smallValue;
            gradients.BiasGradient = largeValue;
            
            // Assert
            Assert.Equal(largeValue, gradients.WeightGradient[0]);
            Assert.Equal(smallValue, gradients.WeightGradient[1]);
            Assert.Equal(largeValue, gradients.BiasGradient);
        }

        [Fact]
        public void Gradients_CanHandleSpecialDoubleValues()
        {
            // Arrange
            var gradients = new Gradients(3);
            
            // Act
            gradients.WeightGradient[0] = double.NaN;
            gradients.WeightGradient[1] = double.PositiveInfinity;
            gradients.WeightGradient[2] = double.NegativeInfinity;
            gradients.BiasGradient = double.NaN;
            
            // Assert
            Assert.True(double.IsNaN(gradients.WeightGradient[0]));
            Assert.True(double.IsPositiveInfinity(gradients.WeightGradient[1]));
            Assert.True(double.IsNegativeInfinity(gradients.WeightGradient[2]));
            Assert.True(double.IsNaN(gradients.BiasGradient));
        }

        #endregion

        #region Typical Usage Scenarios

        [Fact]
        public void Gradients_TypicalBackpropagationScenario_AccumulatesCorrectly()
        {
            // Arrange
            var gradients = new Gradients(3);
            double[] batchGradients1 = { 0.1, 0.2, 0.3 };
            double[] batchGradients2 = { 0.4, 0.5, 0.6 };
            double biasGradient1 = 0.1;
            double biasGradient2 = 0.2;
            
            // Act - Simulate accumulating gradients over multiple samples
            for (int i = 0; i < batchGradients1.Length; i++)
            {
                gradients.WeightGradient[i] += batchGradients1[i];
            }
            gradients.BiasGradient += biasGradient1;
            
            for (int i = 0; i < batchGradients2.Length; i++)
            {
                gradients.WeightGradient[i] += batchGradients2[i];
            }
            gradients.BiasGradient += biasGradient2;
            
            // Assert
            Assert.Equal(0.5, gradients.WeightGradient[0], Tolerance); // 0.1 + 0.4
            Assert.Equal(0.7, gradients.WeightGradient[1], Tolerance); // 0.2 + 0.5
            Assert.Equal(0.9, gradients.WeightGradient[2], Tolerance); // 0.3 + 0.6
            Assert.Equal(0.3, gradients.BiasGradient, Tolerance); // 0.1 + 0.2
        }

        [Fact]
        public void Gradients_ResetScenario_CanBeResetToZero()
        {
            // Arrange
            var gradients = new Gradients(2);
            gradients.WeightGradient[0] = 0.5;
            gradients.WeightGradient[1] = -0.3;
            gradients.BiasGradient = 0.7;
            
            // Act - Reset gradients (as would happen between epochs)
            for (int i = 0; i < gradients.WeightGradient.Length; i++)
            {
                gradients.WeightGradient[i] = 0.0;
            }
            gradients.BiasGradient = 0.0;
            
            // Assert
            Assert.Equal(0.0, gradients.WeightGradient[0]);
            Assert.Equal(0.0, gradients.WeightGradient[1]);
            Assert.Equal(0.0, gradients.BiasGradient);
        }

        [Fact]
        public void Gradients_AverageScenario_CanBeAveragedOverBatchSize()
        {
            // Arrange
            var gradients = new Gradients(2);
            gradients.WeightGradient[0] = 1.0;
            gradients.WeightGradient[1] = 2.0;
            gradients.BiasGradient = 3.0;
            int batchSize = 4;
            
            // Act - Average gradients over batch size
            for (int i = 0; i < gradients.WeightGradient.Length; i++)
            {
                gradients.WeightGradient[i] /= batchSize;
            }
            gradients.BiasGradient /= batchSize;
            
            // Assert
            Assert.Equal(0.25, gradients.WeightGradient[0], Tolerance); // 1.0 / 4
            Assert.Equal(0.5, gradients.WeightGradient[1], Tolerance);  // 2.0 / 4
            Assert.Equal(0.75, gradients.BiasGradient, Tolerance);      // 3.0 / 4
        }

        #endregion

        #region Error Handling Tests

        [Theory]
        [InlineData(1)]
        [InlineData(5)]
        [InlineData(10)]
        [InlineData(100)]
        public void Gradients_Constructor_WithValidWeightCounts_CreatesCorrectArrays(int weightCount)
        {
            // Arrange & Act
            var gradients = new Gradients(weightCount);
            
            // Assert
            Assert.Equal(weightCount, gradients.WeightGradient.Length);
            Assert.All(gradients.WeightGradient, gradient => Assert.Equal(0.0, gradient));
        }

        [Fact]
        public void Gradients_ArrayBoundsAccess_ThrowsExceptionWhenAccessingInvalidIndex()
        {
            // Arrange
            var gradients = new Gradients(3);
            
            // Act & Assert
            Assert.Throws<IndexOutOfRangeException>(() => gradients.WeightGradient[3]);
            // Cannot test negative index due to compiler warning - it's a compile-time error anyway
        }

        #endregion
    }
}
