using System.Diagnostics;
using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class TwoLayerAdder
{
    const int epochs = 20000;
    const double learningRate = 0.001;
    double[][] samples = [];
    double[][] observed = [];
    INeuralNetwork? network;
    NeuralNetworkTrainer? trainer;

    public void CreateTrainingData()
    {
        samples = [[5, 5], [2, 2]];
        observed = [[10], [4]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.Unit, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(2, [2, 1], af);
        network = networkCreator.CreateNetwork();
        trainer = new(network, learningRate);
    }

    internal class TwoLayerNetwork
    {
        internal double diff, loss, dLoss;
        internal const double wInit = 0.01;
        internal double[][][] w = [
            [
                [0.01, 0.02],
                [0.015, 0.014]
            ],
            [
                [0.001, 0.021]
            ]
        ];
        internal double[][][] b = [
            [
                [0],
                [0]
            ],
            [
                [0]
            ]
        ];
        internal double[][] x = [[0, 0], [0]];
        internal double[][] y = [[0, 0], [0]];
        internal double[][][] wGrad = [
            [
                [0, 0], [0, 0]
            ],
            [
                [0, 0]
            ]
        ];
        internal double[][][] wGradSum = [
            [
                [0, 0], [0, 0]
            ],
            [
                [0, 01]
            ]
        ];
        internal double[][][] bGrad = [
            [
                [0, 0], [0, 0]
            ],
            [
                [0, 0]
            ]
        ];
        double[][][] bGradSum = [
            [
                [0, 0], [0, 0]
            ],
            [
                [0, 0]
            ]
        ];


        double[] ProcessLayer0Inputs(double[] i)
        {
            double[] x0 = x[0];
            double[][] w0 = w[0];
            double[][] b0 = b[0];

            x0[0] = i[0] * w0[0][0] + i[1] * w0[0][1] + b0[0][0];
            x0[1] = i[0] * w0[1][0] + i[1] * w0[1][1] + b0[1][0];
            return x0;
        }
        double[] ProcessOutputLayerInputs(double[] i)
        {
            double[] x1 = x[1];
            double[][] w1 = w[1];
            double[][] b1 = b[1];

            x1[0] = i[0] * w1[0][0] + i[1] * w1[0][1] + b1[0][0];

            return x1;
        }

        double[] ActivateLayer0(double[] i)
        {
            double[] y0 = y[0];
            y0[0] = ActivationFunctions.Unit(i[0]);
            y0[1] = ActivationFunctions.Unit(i[1]);
            return y0; ;
        }
        double[] ActivateOutputLayer(double[] i)
        {
            double[] y1 = y[1];
            y1[0] = ActivationFunctions.Unit(i[0]);
            return y1;
        }

        internal double Predict( double[] i)
        {
            x[0] = ProcessLayer0Inputs(i);
            y[0] = ActivateLayer0(x[0]);
            x[1] = ProcessOutputLayerInputs(y[0]);
            y[1] = ActivateOutputLayer(x[1]);
            return y[1][0];
        }
        double ActivateDerivative(double x)
        {
            return ActivationFunctions.UnitDerivative(x);
        }
        void VerifyPrediction(double[] i, double observed)
        {
            Func<double, double> AD = ActivateDerivative;
            double p = Predict(i);

            diff = y[1][0] - observed;
            loss = diff * diff;
            dLoss = 2 * diff;

            bGradSum[1][0][0] += bGrad[1][0][0] = dLoss * AD(x[1][0]);
            wGradSum[1][0][0] += wGrad[1][0][0] = dLoss * AD(x[1][0]) * y[0][0];
            wGradSum[1][0][1] += wGrad[1][0][1] = dLoss * AD(x[1][0]) * y[0][1];

            bGradSum[0][0][0] += bGrad[0][0][0] = dLoss * AD(x[1][0]) * w[1][0][0] * AD(x[0][0]);
            wGradSum[0][0][0] += wGrad[0][0][0] = dLoss * AD(x[1][0]) * w[1][0][0] * AD(x[0][0]) * i[0];
            wGradSum[0][0][1] += wGrad[0][0][1] = dLoss * AD(x[1][0]) * w[1][0][0] * AD(x[0][0]) * i[1];

            bGradSum[0][1][0] += bGrad[0][1][0] = dLoss * AD(x[1][0]) * w[1][0][1] * AD(x[0][1]);
            wGradSum[0][1][0] += wGrad[0][1][0] = dLoss * AD(x[1][0]) * w[1][0][1] * AD(x[0][1]) * i[0];
            wGradSum[0][1][1] += wGrad[0][1][1] = dLoss * AD(x[1][0]) * w[1][0][1] * AD(x[0][1]) * i[1];

            // Console.Write($"wGrad:");
            // for (int j = 0; j < wGrad.Length; j++)
            //     for (int k = 0; k < wGrad[j].Length; k++)
            //         for (int l = 0; l < wGrad[j][k].Length; l++)
            //         {
            //             Console.Write($"[{j},{k},{l}] = {wGrad[j][k][l]} ");
            //         }
            // Console.WriteLine();
        }

        internal void VerifyEpoch(double[][] samples, double[][] observed)
        {
            wGradSum = [[[0, 0], [0, 0]], [[0, 0]]];
            bGradSum = [[[0, 0], [0, 0]], [[0, 0]]];

            VerifyPrediction(samples[0], observed[0][0]);
            // Console.WriteLine($"P0={y[1][0]}");
            VerifyPrediction(samples[1], observed[1][0]);
            // Console.WriteLine($"P1={y[1][0]}");

            b[0][0][0] -= bGradSum[0][0][0] * learningRate / samples.Length; 
            w[0][0][0] -= wGradSum[0][0][0] * learningRate / samples.Length;
            w[0][0][1] -= wGradSum[0][0][1] * learningRate / samples.Length;

            b[0][1][0] -= bGradSum[0][1][0] * learningRate / samples.Length;
            w[0][1][0] -= wGradSum[0][1][0] * learningRate / samples.Length;
            w[0][1][1] -= wGradSum[0][1][1] * learningRate / samples.Length;

            b[1][0][0] -= bGradSum[1][0][0] * learningRate / samples.Length;
            w[1][0][0] -= wGradSum[1][0][0] * learningRate / samples.Length;
            w[1][0][1] -= wGradSum[1][0][1] * learningRate / samples.Length;
        }


    }

    public void Train()
    {
        network!.Weigths[0][0][0] = 0.01;
        network!.Weigths[0][0][1] = 0.02;

        network!.Weigths[0][1][0] = 0.015;
        network!.Weigths[0][1][1] = 0.014;

        network!.Weigths[1][0][0] = 0.001;
        network!.Weigths[1][0][1] = 0.021;

        TwoLayerNetwork epochVerifier = new();
        for (int i = 0; i < epochs; i++)
        {
            trainer!.TrainOneEpoch(samples, observed);
            epochVerifier.VerifyEpoch(samples, observed);

            double[][][] nW = network.Weigths, eW = epochVerifier.w;
            bool same = true;
            for (int ii = 0; ii < nW.Length; ii++)
            {
                for (int j = 0; j < nW[ii].Length; j++)
                {
                    for (int k = 0; k < nW[ii][j].Length; k++)
                    {
                        same &= nW[ii][j][k] == eW[ii][j][k];
                        // Console.Write($"nw[{ii},{j},{k}]={nW[ii][j][k]} ,ew[]={eW[ii][j][k]}  ");

                    }
                    // Console.WriteLine();
                }
                // Console.WriteLine();
            }
            bool different = !same;
            if (different)
            {
                Console.WriteLine("Algorithms differ");
                break;
            }
        }
        var prediction = epochVerifier.Predict([5, 5]);
        Console.WriteLine($"5+5 => {prediction}");
        prediction = epochVerifier.Predict([2,2]);
        Console.WriteLine($"2+2 => {prediction}");
        prediction = epochVerifier.Predict([2, 5]);
        Console.WriteLine($"2+5 => {prediction}");
        prediction = epochVerifier.Predict([5,2]);
        Console.WriteLine($"5+2 => {prediction}");
    }


    private double[][][] FixedWeights()
    {
        double FixedWeight(double _) => 0;
        NetworkCreator.ApplyOn3dArr(network!.Weigths, FixedWeight);
        return network.Weigths;
    }
    public void DoIt()
    {
        CreateNetwork();

        CreateTrainingData();

        Train();


        Console.WriteLine("After training:");
        var prediction = network!.Predict([5,5]);
        Console.WriteLine($"5+5 => {prediction[0]}");
        prediction = network.Predict([2,2]);
        Console.WriteLine($"2+2 => {prediction[0]}");
        prediction = network.Predict([5,2]);
        Console.WriteLine($"5+2 => {prediction[0]}");
    }
}
