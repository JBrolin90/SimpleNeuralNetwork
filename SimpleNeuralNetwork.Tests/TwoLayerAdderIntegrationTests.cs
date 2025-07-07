using Xunit;
using System;
using BackPropagation.NNLib;
using BackPropagation;

namespace SimpleNeuralNetwork.Tests
{
    public class TwoLayerAdderIntegrationTests
    {
        private const double Tolerance = 1e-6;
        private const int Epochs = 20000;
        private const double LearningRate = 0.001;

        [Fact]
        public void TwoLayerAdder_TrainAndPredict_ConvergesToExpectedResults()
        {
            // Arrange
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();
            twoLayerAdderTest.CreateTrainingData();

            // Act
            twoLayerAdderTest.Train();

            // Assert - Test predictions after training
            var prediction1 = twoLayerAdderTest.Network!.Predict(new double[] { 5, 5 });
            var prediction2 = twoLayerAdderTest.Network!.Predict(new double[] { 2, 2 });
            var prediction3 = twoLayerAdderTest.Network!.Predict(new double[] { 5, 2 });
            var prediction4 = twoLayerAdderTest.Network!.Predict(new double[] { 2, 5 });

            // The network should learn to add the inputs accurately
            Assert.True(Math.Abs(prediction1[0] - 10.0) < 0.01, $"Expected ~10, got {prediction1[0]}");
            Assert.True(Math.Abs(prediction2[0] - 4.0) < 0.01, $"Expected ~4, got {prediction2[0]}");
            Assert.True(Math.Abs(prediction3[0] - 7.0) < 0.1, $"Expected ~7, got {prediction3[0]}");
            Assert.True(Math.Abs(prediction4[0] - 7.0) < 0.1, $"Expected ~7, got {prediction4[0]}");
        }

        [Fact]
        public void TwoLayerAdder_NetworkCreation_InitializesCorrectly()
        {
            // Arrange & Act
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();

            // Assert
            Assert.NotNull(twoLayerAdderTest.Network);
            Assert.NotNull(twoLayerAdderTest.Trainer);
            Assert.Equal(2, twoLayerAdderTest.Network.Layers.Length);
            
            // First layer should have 2 neurons (hidden layer)
            Assert.Equal(2, twoLayerAdderTest.Network.Layers[0].Neurons.Length);
            
            // Second layer should have 1 neuron (output layer)
            Assert.Single(twoLayerAdderTest.Network.Layers[1].Neurons);

            // Check weight initialization
            Assert.Equal(2, twoLayerAdderTest.Network.Weigths.Length); // 2 layers
            Assert.Equal(2, twoLayerAdderTest.Network.Weigths[0].Length); // 2 neurons in first layer
            Assert.Equal(2, twoLayerAdderTest.Network.Weigths[0][0].Length); // 2 inputs to first neuron
            Assert.Single(twoLayerAdderTest.Network.Weigths[1]); // 1 neuron in second layer
            Assert.Equal(2, twoLayerAdderTest.Network.Weigths[1][0].Length); // 2 inputs from first layer
        }

        [Fact]
        public void TwoLayerAdder_TrainingData_CreatedCorrectly()
        {
            // Arrange & Act
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateTrainingData();

            // Assert
            Assert.Equal(2, twoLayerAdderTest.Samples.Length);
            Assert.Equal(2, twoLayerAdderTest.Observed.Length);

            // Check training samples
            Assert.Equal(5.0, twoLayerAdderTest.Samples[0][0]);
            Assert.Equal(5.0, twoLayerAdderTest.Samples[0][1]);
            Assert.Equal(2.0, twoLayerAdderTest.Samples[1][0]);
            Assert.Equal(2.0, twoLayerAdderTest.Samples[1][1]);

            // Check expected outputs
            Assert.Equal(10.0, twoLayerAdderTest.Observed[0][0]);
            Assert.Equal(4.0, twoLayerAdderTest.Observed[1][0]);
        }

        [Fact]
        public void TwoLayerAdder_InitialWeights_SetCorrectly()
        {
            // Arrange
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();
            twoLayerAdderTest.CreateTrainingData();

            // Act
            twoLayerAdderTest.SetInitialWeights();

            // Assert - Check that initial weights are set to expected values
            Assert.Equal(0.01, twoLayerAdderTest.Network!.Weigths[0][0][0]);
            Assert.Equal(0.02, twoLayerAdderTest.Network!.Weigths[0][0][1]);
            Assert.Equal(0.015, twoLayerAdderTest.Network!.Weigths[0][1][0]);
            Assert.Equal(0.014, twoLayerAdderTest.Network!.Weigths[0][1][1]);
            Assert.Equal(0.001, twoLayerAdderTest.Network!.Weigths[1][0][0]);
            Assert.Equal(0.021, twoLayerAdderTest.Network!.Weigths[1][0][1]);
        }

        [Fact]
        public void TwoLayerAdder_TrainingProcess_ConvergesOverTime()
        {
            // Arrange
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();
            twoLayerAdderTest.CreateTrainingData();
            twoLayerAdderTest.SetInitialWeights();

            // Get initial predictions
            var initialPrediction1 = twoLayerAdderTest.Network!.Predict(new double[] { 5, 5 });
            var initialPrediction2 = twoLayerAdderTest.Network!.Predict(new double[] { 2, 2 });

            // Calculate initial error
            double initialError1 = Math.Abs(initialPrediction1[0] - 10.0);
            double initialError2 = Math.Abs(initialPrediction2[0] - 4.0);
            double initialTotalError = initialError1 + initialError2;

            // Act
            twoLayerAdderTest.Train();

            // Get final predictions
            var finalPrediction1 = twoLayerAdderTest.Network!.Predict(new double[] { 5, 5 });
            var finalPrediction2 = twoLayerAdderTest.Network!.Predict(new double[] { 2, 2 });

            // Calculate final error
            double finalError1 = Math.Abs(finalPrediction1[0] - 10.0);
            double finalError2 = Math.Abs(finalPrediction2[0] - 4.0);
            double finalTotalError = finalError1 + finalError2;

            // Assert that training improved the predictions
            Assert.True(finalTotalError < initialTotalError, 
                $"Training should improve predictions. Initial error: {initialTotalError}, Final error: {finalTotalError}");
            
            // Assert that final error is very small
            Assert.True(finalTotalError < 0.1, $"Final error should be small: {finalTotalError}");
        }

        [Fact]
        public void TwoLayerAdder_WeightVerification_MatchesManualCalculation()
        {
            // Arrange
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();
            twoLayerAdderTest.CreateTrainingData();

            // Act
            bool weightsMatch = twoLayerAdderTest.TrainWithVerification(100); // Train for fewer epochs for this test

            // Assert
            Assert.True(weightsMatch, "Network weights should match manual calculation during training");
        }

        [Theory]
        [InlineData(new double[] { 5, 5 }, 10.0)]
        [InlineData(new double[] { 2, 2 }, 4.0)]
        [InlineData(new double[] { 3, 3 }, 6.0)]
        [InlineData(new double[] { 1, 1 }, 2.0)]
        public void TwoLayerAdder_PredictAfterTraining_ReturnsExpectedValues(double[] inputs, double expected)
        {
            // Arrange
            var twoLayerAdderTest = new TwoLayerAdderTestWrapper();
            twoLayerAdderTest.CreateNetwork();
            twoLayerAdderTest.CreateTrainingData();

            // Act
            twoLayerAdderTest.Train();
            var prediction = twoLayerAdderTest.Network!.Predict(inputs);

            // Assert
            Assert.True(Math.Abs(prediction[0] - expected) < 0.2, 
                $"Expected {expected}, got {prediction[0]} for input [{inputs[0]}, {inputs[1]}]");
        }
    }

    // Wrapper class that mimics TwoLayerAdder behavior for testing
    public class TwoLayerAdderTestWrapper
    {
        private const int Epochs = 20000;
        private const double LearningRate = 0.001;
        
        public double[][] Samples { get; private set; } = Array.Empty<double[]>();
        public double[][] Observed { get; private set; } = Array.Empty<double[]>();
        public INeuralNetwork? Network { get; private set; }
        public NeuralNetworkTrainer? Trainer { get; private set; }

        public void CreateTrainingData()
        {
            Samples = new double[][] { new double[] { 5, 5 }, new double[] { 2, 2 } };
            Observed = new double[][] { new double[] { 10 }, new double[] { 4 } };
        }

        public void CreateNetwork()
        {
            Func<double, double>[] af = new Func<double, double>[] { ActivationFunctions.Unit, ActivationFunctions.Unit };
            var networkCreator = new NetworkCreator(2, new int[] { 2, 1 }, af);
            Network = networkCreator.CreateNetwork();
            Trainer = new NeuralNetworkTrainer(Network, LearningRate);
        }

        public void SetInitialWeights()
        {
            if (Network == null) return;

            // Set the specific initial weights as used in the original TwoLayerAdder
            Network.Weigths[0][0][0] = 0.01;
            Network.Weigths[0][0][1] = 0.02;
            Network.Weigths[0][1][0] = 0.015;
            Network.Weigths[0][1][1] = 0.014;
            Network.Weigths[1][0][0] = 0.001;
            Network.Weigths[1][0][1] = 0.021;
        }

        public void Train()
        {
            SetInitialWeights();
            
            var epochVerifier = new TwoLayerEpochVerifierWrapper();
            for (int i = 0; i < Epochs; i++)
            {
                Trainer!.TrainOneEpoch(Samples, Observed);
                epochVerifier.VerifyEpoch(Samples, Observed);

                // Check if algorithms match (like the original implementation)
                double[][][] nW = Network!.Weigths, eW = epochVerifier.W;
                bool same = true;
                for (int ii = 0; ii < nW.Length; ii++)
                {
                    for (int j = 0; j < nW[ii].Length; j++)
                    {
                        for (int k = 0; k < nW[ii][j].Length; k++)
                        {
                            same &= Math.Abs(nW[ii][j][k] - eW[ii][j][k]) < 1e-10;
                        }
                    }
                }
                
                if (!same)
                {
                    break; // Algorithms differ, stop training
                }
            }
        }

        public bool TrainWithVerification(int epochs)
        {
            SetInitialWeights();
            
            var epochVerifier = new TwoLayerEpochVerifierWrapper();
            for (int i = 0; i < epochs; i++)
            {
                Trainer!.TrainOneEpoch(Samples, Observed);
                epochVerifier.VerifyEpoch(Samples, Observed);

                // Check if algorithms match
                double[][][] nW = Network!.Weigths, eW = epochVerifier.W;
                bool same = true;
                for (int ii = 0; ii < nW.Length; ii++)
                {
                    for (int j = 0; j < nW[ii].Length; j++)
                    {
                        for (int k = 0; k < nW[ii][j].Length; k++)
                        {
                            same &= Math.Abs(nW[ii][j][k] - eW[ii][j][k]) < 1e-10;
                        }
                    }
                }
                
                if (!same)
                {
                    return false;
                }
            }
            return true;
        }
    }

    // Wrapper class that mimics the TwoLayerNetwork verification logic
    public class TwoLayerEpochVerifierWrapper
    {
        public double[][][] W { get; set; }
        public double[][][] B { get; set; }
        
        private double diff, loss, dLoss;
        private double[][] x = new double[][] { new double[2], new double[1] };
        private double[][] y = new double[][] { new double[2], new double[1] };
        private double[][][] wGrad;
        private double[][][] wGradSum;
        private double[][][] bGrad;
        private double[][][] bGradSum;
        private const double LearningRate = 0.001;

        public TwoLayerEpochVerifierWrapper()
        {
            // Initialize weights like the original TwoLayerNetwork
            W = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0.01, 0.02 },
                    new double[] { 0.015, 0.014 }
                },
                new double[][]
                {
                    new double[] { 0.001, 0.021 }
                }
            };

            B = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0 },
                    new double[] { 0 }
                },
                new double[][]
                {
                    new double[] { 0 }
                }
            };

            wGrad = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 0 }
                },
                new double[][]
                {
                    new double[] { 0, 0 }
                }
            };

            wGradSum = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0, 0 },
                    new double[] { 0, 0 }
                },
                new double[][]
                {
                    new double[] { 0, 0 }
                }
            };

            bGrad = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0 },
                    new double[] { 0 }
                },
                new double[][]
                {
                    new double[] { 0 }
                }
            };

            bGradSum = new double[][][]
            {
                new double[][]
                {
                    new double[] { 0 },
                    new double[] { 0 }
                },
                new double[][]
                {
                    new double[] { 0 }
                }
            };
        }

        private double[] ProcessLayer0Inputs(double[] i)
        {
            double[] x0 = x[0];
            double[][] w0 = W[0];
            double[][] b0 = B[0];

            x0[0] = i[0] * w0[0][0] + i[1] * w0[0][1] + b0[0][0];
            x0[1] = i[0] * w0[1][0] + i[1] * w0[1][1] + b0[1][0];
            return x0;
        }

        private double[] ProcessOutputLayerInputs(double[] i)
        {
            double[] x1 = x[1];
            double[][] w1 = W[1];
            double[][] b1 = B[1];

            x1[0] = i[0] * w1[0][0] + i[1] * w1[0][1] + b1[0][0];
            return x1;
        }

        private double[] ActivateLayer0(double[] i)
        {
            double[] y0 = y[0];
            y0[0] = ActivationFunctions.Unit(i[0]);
            y0[1] = ActivationFunctions.Unit(i[1]);
            return y0;
        }

        private double[] ActivateOutputLayer(double[] i)
        {
            double[] y1 = y[1];
            y1[0] = ActivationFunctions.Unit(i[0]);
            return y1;
        }

        private double Predict(double[] i)
        {
            x[0] = ProcessLayer0Inputs(i);
            y[0] = ActivateLayer0(x[0]);
            x[1] = ProcessOutputLayerInputs(y[0]);
            y[1] = ActivateOutputLayer(x[1]);
            return y[1][0];
        }

        private double ActivateDerivative(double x)
        {
            return ActivationFunctions.UnitDerivative(x);
        }

        private void VerifyPrediction(double[] i, double observed)
        {
            Func<double, double> AD = ActivateDerivative;
            double p = Predict(i);

            diff = y[1][0] - observed;
            loss = diff * diff;
            dLoss = 2 * diff;

            bGradSum[1][0][0] += bGrad[1][0][0] = dLoss * AD(x[1][0]);
            wGradSum[1][0][0] += wGrad[1][0][0] = dLoss * AD(x[1][0]) * y[0][0];
            wGradSum[1][0][1] += wGrad[1][0][1] = dLoss * AD(x[1][0]) * y[0][1];

            bGradSum[0][0][0] += bGrad[0][0][0] = dLoss * AD(x[1][0]) * W[1][0][0] * AD(x[0][0]);
            wGradSum[0][0][0] += wGrad[0][0][0] = dLoss * AD(x[1][0]) * W[1][0][0] * AD(x[0][0]) * i[0];
            wGradSum[0][0][1] += wGrad[0][0][1] = dLoss * AD(x[1][0]) * W[1][0][0] * AD(x[0][0]) * i[1];

            bGradSum[0][1][0] += bGrad[0][1][0] = dLoss * AD(x[1][0]) * W[1][0][1] * AD(x[0][1]);
            wGradSum[0][1][0] += wGrad[0][1][0] = dLoss * AD(x[1][0]) * W[1][0][1] * AD(x[0][1]) * i[0];
            wGradSum[0][1][1] += wGrad[0][1][1] = dLoss * AD(x[1][0]) * W[1][0][1] * AD(x[0][1]) * i[1];
        }

        public void VerifyEpoch(double[][] samples, double[][] observed)
        {
            // Reset gradients
            for (int i = 0; i < wGradSum.Length; i++)
            {
                for (int j = 0; j < wGradSum[i].Length; j++)
                {
                    for (int k = 0; k < wGradSum[i][j].Length; k++)
                    {
                        wGradSum[i][j][k] = 0;
                    }
                }
            }

            for (int i = 0; i < bGradSum.Length; i++)
            {
                for (int j = 0; j < bGradSum[i].Length; j++)
                {
                    for (int k = 0; k < bGradSum[i][j].Length; k++)
                    {
                        bGradSum[i][j][k] = 0;
                    }
                }
            }

            VerifyPrediction(samples[0], observed[0][0]);
            VerifyPrediction(samples[1], observed[1][0]);

            // Update weights and biases
            B[0][0][0] -= bGradSum[0][0][0] * LearningRate / samples.Length;
            W[0][0][0] -= wGradSum[0][0][0] * LearningRate / samples.Length;
            W[0][0][1] -= wGradSum[0][0][1] * LearningRate / samples.Length;

            B[0][1][0] -= bGradSum[0][1][0] * LearningRate / samples.Length;
            W[0][1][0] -= wGradSum[0][1][0] * LearningRate / samples.Length;
            W[0][1][1] -= wGradSum[0][1][1] * LearningRate / samples.Length;

            B[1][0][0] -= bGradSum[1][0][0] * LearningRate / samples.Length;
            W[1][0][0] -= wGradSum[1][0][0] * LearningRate / samples.Length;
            W[1][0][1] -= wGradSum[1][0][1] * LearningRate / samples.Length;
        }
    }
}
