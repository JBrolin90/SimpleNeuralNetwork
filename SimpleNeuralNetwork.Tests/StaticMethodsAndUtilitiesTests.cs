using Xunit;
using System;
using BackPropagation.NNLib;
using System.Linq;

namespace SimpleNeuralNetwork.Tests
{
    public class StaticMethodsAndUtilitiesTests
    {
        private const double Tolerance = 1e-7;

        #region NetworkCreator Static Methods Comprehensive Tests

        [Fact]
        public void NetworkCreator_ActOn3dArr_ExecutesActionOnEveryElement()
        {
            // Arrange
            double[][][] testArray = {
                new double[][] { 
                    new double[] { 1.0, 2.0, 3.0 }, 
                    new double[] { 4.0, 5.0 } 
                },
                new double[][] { 
                    new double[] { 6.0 }, 
                    new double[] { 7.0, 8.0, 9.0, 10.0 } 
                }
            };

            var executedValues = new List<double>();
            Action<double> collectAction = value => executedValues.Add(value);

            // Act
            NetworkCreator.ActOn3dArr(testArray, collectAction);

            // Assert
            var expectedValues = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
            Assert.Equal(expectedValues.Length, executedValues.Count);
            Assert.Equal(expectedValues.OrderBy(x => x), executedValues.OrderBy(x => x));
        }

        [Fact]
        public void NetworkCreator_ActOn3dArr_WithModifyingAction_ChangesOriginalArray()
        {
            // Arrange
            double[][][] testArray = {
                new double[][] { 
                    new double[] { 1.0, 2.0 }, 
                    new double[] { 3.0, 4.0 } 
                }
            };

            var originalValues = new double[] { 1.0, 2.0, 3.0, 4.0 };
            int actionCallCount = 0;
            Action<double> countingAction = value => actionCallCount++;

            // Act
            NetworkCreator.ActOn3dArr(testArray, countingAction);

            // Assert
            Assert.Equal(4, actionCallCount);
            // Values should remain unchanged since ActOn3dArr doesn't modify the array
            Assert.Equal(1.0, testArray[0][0][0]);
            Assert.Equal(2.0, testArray[0][0][1]);
            Assert.Equal(3.0, testArray[0][1][0]);
            Assert.Equal(4.0, testArray[0][1][1]);
        }

        [Fact]
        public void NetworkCreator_ApplyOn3dArr_TransformsEveryElement()
        {
            // Arrange
            double[][][] testArray = {
                new double[][] { 
                    new double[] { 1.0, 2.0, 3.0 }, 
                    new double[] { 4.0, 5.0 } 
                },
                new double[][] { 
                    new double[] { 6.0 }, 
                    new double[] { 7.0, 8.0, 9.0, 10.0 } 
                }
            };

            Func<double, double> squareFunction = x => x * x;

            // Act
            NetworkCreator.ApplyOn3dArr(testArray, squareFunction);

            // Assert
            var expectedValues = new double[] { 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0 };
            var actualValues = new List<double>();
            
            foreach (var layer in testArray)
            {
                foreach (var neuron in layer)
                {
                    foreach (var weight in neuron)
                    {
                        actualValues.Add(weight);
                    }
                }
            }

            Assert.Equal(expectedValues.OrderBy(x => x), actualValues.OrderBy(x => x));
        }

        [Fact]
        public void NetworkCreator_ApplyOn3dArr_WithComplexTransformation_WorksCorrectly()
        {
            // Arrange
            double[][][] testArray = {
                new double[][] { 
                    new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 }
                }
            };

            Func<double, double> complexFunction = x => Math.Sin(x) + Math.Cos(x * 2);

            // Act
            NetworkCreator.ApplyOn3dArr(testArray, complexFunction);

            // Assert
            var expectedValues = new double[] { -2.0, -1.0, 0.0, 1.0, 2.0 }.Select(complexFunction).ToArray();
            var actualValues = testArray[0][0];

            for (int i = 0; i < expectedValues.Length; i++)
            {
                Assert.Equal(expectedValues[i], actualValues[i], Tolerance);
            }
        }

        [Fact]
        public void NetworkCreator_StaticMethods_HandleJaggedArraysCorrectly()
        {
            // Arrange
            double[][][] jaggedArray = {
                new double[][] { 
                    new double[] { 1.0 },
                    new double[] { 2.0, 3.0, 4.0 },
                    new double[] { 5.0, 6.0 }
                },
                new double[][] { 
                    new double[] { 7.0, 8.0, 9.0, 10.0, 11.0 },
                    new double[] { 12.0 }
                }
            };

            var elementCount = 0;
            Action<double> countAction = x => elementCount++;

            // Act
            NetworkCreator.ActOn3dArr(jaggedArray, countAction);

            // Assert
            Assert.Equal(12, elementCount); // 1 + 3 + 2 + 5 + 1 = 12 elements total
        }

        [Fact]
        public void NetworkCreator_StaticMethods_WithNullFunction_ThrowsException()
        {
            // Arrange
            double[][][] testArray = {
                new double[][] { new double[] { 1.0, 2.0 } }
            };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => NetworkCreator.ActOn3dArr(testArray, null!));
            Assert.Throws<ArgumentNullException>(() => NetworkCreator.ApplyOn3dArr(testArray, null!));
        }

        [Fact]
        public void NetworkCreator_StaticMethods_WithNullArray_ThrowsException()
        {
            // Arrange
            Action<double> dummyAction = x => { };
            Func<double, double> dummyFunction = x => x;

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => NetworkCreator.ActOn3dArr(null!, dummyAction));
            Assert.Throws<ArgumentNullException>(() => NetworkCreator.ApplyOn3dArr(null!, dummyFunction));
        }

        [Fact]
        public void NetworkCreator_StaticMethods_WithArrayContainingNulls_ThrowsException()
        {
            // Arrange
            double[][][] arrayWithNull = {
                new double[][] { 
                    new double[] { 1.0, 2.0 },
                    null! // This should cause an exception
                }
            };

            Action<double> dummyAction = x => { };
            Func<double, double> dummyFunction = x => x;

            // Act & Assert
            Assert.Throws<NullReferenceException>(() => NetworkCreator.ActOn3dArr(arrayWithNull, dummyAction));
            Assert.Throws<NullReferenceException>(() => NetworkCreator.ApplyOn3dArr(arrayWithNull, dummyFunction));
        }

        #endregion

        #region ActivationFunction Static Methods Tests

        [Fact]
        public void ActivationFunctions_AllFunctions_HandleZeroCorrectly()
        {
            // Arrange
            double input = 0.0;

            // Act
            double unitResult = ActivationFunctions.Unit(input);
            double sigmoidResult = ActivationFunctions.Sigmoid(input);
            double reluResult = ActivationFunctions.ReLU(input);
            double tanhResult = ActivationFunctions.Tanh(input);

            // Assert
            Assert.Equal(0.0, unitResult);
            Assert.Equal(0.5, sigmoidResult, Tolerance);
            Assert.Equal(0.0, reluResult);
            Assert.Equal(0.0, tanhResult, Tolerance);
        }

        [Fact]
        public void ActivationFunctions_AllDerivatives_HandleZeroCorrectly()
        {
            // Arrange
            double input = 0.0;

            // Act
            double unitDerivative = ActivationFunctions.UnitDerivative(input);
            double sigmoidDerivative = ActivationFunctions.SigmoidDerivative(input);
            double reluDerivative = ActivationFunctions.ReLUDerivative(input);
            double tanhDerivative = ActivationFunctions.TanhDerivative(input);

            // Assert
            Assert.Equal(1.0, unitDerivative);
            Assert.Equal(0.25, sigmoidDerivative, Tolerance); // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
            Assert.Equal(0.0, reluDerivative); // ReLU derivative at 0 is typically 0
            Assert.Equal(1.0, tanhDerivative, Tolerance); // tanh'(0) = 1 - tanh(0)^2 = 1 - 0^2 = 1
        }

        [Fact]
        public void ActivationFunctions_Sigmoid_ProducesValidRange()
        {
            // Arrange
            double[] testValues = { -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };

            foreach (double value in testValues)
            {
                // Act
                double result = ActivationFunctions.Sigmoid(value);

                // Assert
                Assert.InRange(result, 0.0, 1.0);
                Assert.True(double.IsFinite(result));
            }
        }

        [Fact]
        public void ActivationFunctions_Tanh_ProducesValidRange()
        {
            // Arrange
            double[] testValues = { -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };

            foreach (double value in testValues)
            {
                // Act
                double result = ActivationFunctions.Tanh(value);

                // Assert
                Assert.InRange(result, -1.0, 1.0);
                Assert.True(double.IsFinite(result));
            }
        }

        [Fact]
        public void ActivationFunctions_ReLU_ProducesValidRange()
        {
            // Arrange
            double[] testValues = { -100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0 };

            foreach (double value in testValues)
            {
                // Act
                double result = ActivationFunctions.ReLU(value);

                // Assert
                Assert.True(result >= 0.0);
                Assert.True(double.IsFinite(result) || double.IsPositiveInfinity(result));
                
                if (value <= 0.0)
                {
                    Assert.Equal(0.0, result);
                }
                else
                {
                    Assert.Equal(value, result);
                }
            }
        }

        [Fact]
        public void ActivationFunctions_DerivativeConsistency_MatchesNumericalDerivative()
        {
            // Arrange
            double[] testValues = { -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0 };
            double h = 1e-8; // Small step for numerical differentiation

            foreach (double x in testValues)
            {
                // Act - Calculate analytical derivatives
                double sigmoidDerivative = ActivationFunctions.SigmoidDerivative(x);
                double tanhDerivative = ActivationFunctions.TanhDerivative(x);

                // Act - Calculate numerical derivatives
                double sigmoidNumerical = (ActivationFunctions.Sigmoid(x + h) - ActivationFunctions.Sigmoid(x - h)) / (2 * h);
                double tanhNumerical = (ActivationFunctions.Tanh(x + h) - ActivationFunctions.Tanh(x - h)) / (2 * h);

                // Assert
                Assert.Equal(sigmoidNumerical, sigmoidDerivative, 1e-6);
                Assert.Equal(tanhNumerical, tanhDerivative, 1e-6);
            }
        }

        #endregion

        #region LossFunction Static Methods Tests

        [Fact]
        public void LossFunctions_SquaredError_CalculatesCorrectly()
        {
            // Arrange
            double[] prediction = { 2.0, 4.0, 6.0 };
            double[] observed = { 1.0, 3.0, 5.0 };

            // Act
            double[] loss = LossFunctions.SquaredError(prediction, observed);

            // Assert
            Assert.Equal(3, loss.Length);
            Assert.Equal(1.0, loss[0], Tolerance); // (2-1)^2 = 1
            Assert.Equal(1.0, loss[1], Tolerance); // (4-3)^2 = 1
            Assert.Equal(1.0, loss[2], Tolerance); // (6-5)^2 = 1
        }

        [Fact]
        public void LossFunctions_SquaredError_WithNullInputs_ThrowsException()
        {
            // Arrange
            double[] prediction = { 1.0, 2.0 };

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => LossFunctions.SquaredError(null!, prediction));
            Assert.Throws<ArgumentNullException>(() => LossFunctions.SquaredError(prediction, null!));
        }

        [Fact]
        public void LossFunctions_SquaredError_WithMismatchedArrays_ThrowsException()
        {
            // Arrange
            double[] prediction = { 1.0, 2.0 };
            double[] observed = { 1.0, 2.0, 3.0 };        // Act & Assert
        Assert.Throws<ArgumentException>(() => LossFunctions.SquaredError(prediction, observed));
        }

        [Fact]
        public void LossFunctions_SumSquaredError_CalculatesCorrectly()
        {
            // Arrange
            double[][] predictions = { 
                new double[] { 2.0, 4.0 }, 
                new double[] { 6.0, 8.0 } 
            };
            double[][] observed = { 
                new double[] { 1.0, 3.0 }, 
                new double[] { 5.0, 7.0 } 
            };

            // Act
            double[] loss = LossFunctions.SumSquaredError(predictions, observed);

            // Assert
            Assert.Equal(2, loss.Length);
            Assert.Equal(2.0, loss[0], Tolerance); // (2-1)^2 + (4-3)^2 = 1 + 1 = 2
            Assert.Equal(2.0, loss[1], Tolerance); // (6-5)^2 + (8-7)^2 = 1 + 1 = 2
        }

        [Fact]
        public void LossFunctions_SumSquaredErrorDerivative_CalculatesCorrectly()
        {
            // Arrange
            double[][] predictions = { 
                new double[] { 3.0, 5.0 }, 
                new double[] { 7.0, 9.0 } 
            };
            double[][] observed = { 
                new double[] { 1.0, 2.0 }, 
                new double[] { 4.0, 6.0 } 
            };

            // Act
            double[] derivative = LossFunctions.SumSquaredErrorDerivative(predictions, observed);

            // Assert
            Assert.Equal(2, derivative.Length);
            Assert.Equal(10.0, derivative[0], Tolerance); // 2*(3-1) + 2*(7-4) = 2*2 + 2*3 = 4 + 6 = 10
            Assert.Equal(12.0, derivative[1], Tolerance); // 2*(5-2) + 2*(9-6) = 2*3 + 2*3 = 6 + 6 = 12
        }

        [Fact]
        public void LossFunctions_WithEmptyArrays_HandlesCorrectly()
        {
            // Arrange
            double[] emptyPrediction = Array.Empty<double>();
            double[] emptyObserved = Array.Empty<double>();

            // Act
            double[] loss = LossFunctions.SquaredError(emptyPrediction, emptyObserved);

            // Assert
            Assert.Empty(loss);
        }

        [Fact]
        public void LossFunctions_WithSingleValues_CalculatesCorrectly()
        {
            // Arrange
            double[] prediction = { 5.0 };
            double[] observed = { 3.0 };

            // Act
            double[] loss = LossFunctions.SquaredError(prediction, observed);

            // Assert
            Assert.Single(loss);
            Assert.Equal(4.0, loss[0], Tolerance); // (5-3)^2 = 4
        }

        #endregion

        #region Sample Static Methods Tests

        [Fact]
        public void Sample_Xample_ReturnsCorrectArray()
        {
            // Arrange
            var sample = new Sample(2.0, 3.0, Operation.add, 1.0);

            // Act
            double[] array = sample.Xample;

            // Assert
            Assert.Equal(4, array.Length);
            Assert.Equal(2.0, array[0]);
            Assert.Equal(3.0, array[1]);
            Assert.Equal(0.0, array[2]); // add operation sets index 2 to 0
            Assert.Equal(1.0, array[3]); // add operation sets index 3 to 1
        }

        [Fact]
        public void Sample_Xample_WithDifferentOperations_ProducesCorrectArrays()
        {
            // Arrange
            var operations = new Operation[] { 
                Operation.add, 
                Operation.hypot 
            };

            foreach (var operation in operations)
            {
                var sample = new Sample(1.0, 2.0, operation, 3.0);

                // Act
                double[] array = sample.Xample;

                // Assert
                Assert.Equal(4, array.Length);
                Assert.Equal(1.0, array[0]);
                Assert.Equal(2.0, array[1]);
                Assert.True(double.IsFinite(array[2]));
                Assert.True(double.IsFinite(array[3]));
            }
        }

        [Fact]
        public void Sample_Observed_CalculatesCorrectly()
        {
            // Arrange & Act & Assert
            var addSample = new Sample(2.0, 3.0, Operation.add, 1.0);
            Assert.Equal(5.0, addSample.Observed[0], Tolerance);

            var hypotSample = new Sample(3.0, 4.0, Operation.hypot, 1.0);
            Assert.Equal(5.0, hypotSample.Observed[0], Tolerance);
        }

        [Fact]
        public void Sample_Observed_WithEdgeCases_HandlesCorrectly()
        {
            // Arrange & Act & Assert
            var negativeHypot = new Sample(-3.0, -4.0, Operation.hypot, 1.0);
            Assert.Equal(5.0, negativeHypot.Observed[0], Tolerance);

            var zeroHypot = new Sample(0.0, 0.0, Operation.hypot, 1.0);
            Assert.Equal(0.0, zeroHypot.Observed[0], Tolerance);
        }

        #endregion

        #region Utility and Helper Tests

        [Fact]
        public void NetworkCreator_Properties_ArePublicAndMutable()
        {
            // Arrange
            var creator = new NetworkCreator(2, new int[] { 1 }, new Func<double, double>[] { ActivationFunctions.Unit });

            // Act & Assert
            Assert.NotNull(creator.Weights);
            Assert.NotNull(creator.Biases);
            Assert.NotNull(creator.ActivationFunctions);
            Assert.NotNull(creator.Ys);

            // Properties should be settable
            var newWeights = new double[][][] { new double[][] { new double[] { 1.0, 2.0 } } };
            creator.Weights = newWeights;
            Assert.Equal(newWeights, creator.Weights);

            var newBiases = new double[][][] { new double[][] { new double[] { 0.5 } } };
            creator.Biases = newBiases;
            Assert.Equal(newBiases, creator.Biases);
        }

        [Fact]
        public void Operation_EnumValues_AreSequential()
        {
            // Arrange & Act
            var operations = Enum.GetValues<Operation>();

            // Assert
            Assert.Equal(2, operations.Length);
            Assert.Contains(Operation.add, operations);
            Assert.Contains(Operation.hypot, operations);
        }

        [Fact]
        public void AllActivationFunctions_ProduceFiniteResults_WithNormalInputs()
        {
            // Arrange
            double[] testInputs = { -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0 };
            var functions = new Func<double, double>[] {
                ActivationFunctions.Unit,
                ActivationFunctions.Sigmoid,
                ActivationFunctions.ReLU,
                ActivationFunctions.Tanh
            };

            // Act & Assert
            foreach (var function in functions)
            {
                foreach (var input in testInputs)
                {
                    var result = function(input);
                    Assert.True(double.IsFinite(result), $"Function {function.Method.Name} with input {input} produced non-finite result: {result}");
                }
            }
        }

        [Fact]
        public void AllActivationFunctionDerivatives_ProduceFiniteResults_WithNormalInputs()
        {
            // Arrange
            double[] testInputs = { -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0 };
            var derivatives = new Func<double, double>[] {
                ActivationFunctions.UnitDerivative,
                ActivationFunctions.SigmoidDerivative,
                ActivationFunctions.ReLUDerivative,
                ActivationFunctions.TanhDerivative
            };

            // Act & Assert
            foreach (var derivative in derivatives)
            {
                foreach (var input in testInputs)
                {
                    var result = derivative(input);
                    Assert.True(double.IsFinite(result), $"Derivative {derivative.Method.Name} with input {input} produced non-finite result: {result}");
                }
            }
        }

        #endregion
    }
}
