using System.Diagnostics;
using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class Linear2LayersTest
{
    const int epochs = 20000;
    const double learningRate = 0.1;
    double[][] samples = [];
    double[][] observed = [];
    NeuralNetworkTrainer? network;

    public void CreateTrainingData()
    {
        samples = [[0], [1]];
        observed = [[0.3], [0.6]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.Unit, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [1, 1], af);
        network = networkCreator.CreateNetwork(learningRate);
    }

    internal class TwoLayerNetwork
    {
        internal double diff, loss, dLoss;
        internal double[] w = [0.01, 0.01], b = [0, 0], x = [0, 0], y = [0, 0];
        double[] wGrad = [0, 0], bGrad = [0, 0], wGradSum = [0, 0], bGradSum = [0, 0];

        double ProcessInput(int layer, double i)
        {
            return x[layer] = i * w[layer] + b[layer];
        }

        double Activate(int layer, double x)
        {
            return y[layer] = x;
        }

        internal double Predict( double i)
        {
            x[0] = ProcessInput(0, i);
            y[0] = Activate(0, x[0]);
            x[1] = ProcessInput(1, y[0]);
            y[1] = Activate(1, x[1]);
            return y[1];
        }
        double ActivateDerivative(double x)
        {
            return 1;
        }
        void VerifyPrediction(double i, double observed)
        {
            y[1] = Predict(i);

            diff = y[1] - observed;
            loss = diff * diff;
            dLoss = 2 * diff;
            wGradSum[1] += wGrad[1] = dLoss * ActivateDerivative(x[1]) * y[0];
            bGradSum[1] += bGrad[1] = dLoss * ActivateDerivative(x[1]);
            wGradSum[0] += wGrad[0] = dLoss * ActivateDerivative(x[1]) * w[1] * ActivateDerivative(x[0]) * i;
            bGradSum[0] += bGrad[0] = dLoss * ActivateDerivative(x[1]) * w[1] * ActivateDerivative(x[0]);
        }

        internal void VerifyEpoch(double[][] samples, double[][] observed)
        {
            wGradSum = [0, 0];
            bGradSum = [0, 0];

            VerifyPrediction(samples[0][0], observed[0][0]);
            VerifyPrediction(samples[1][0], observed[1][0]);

            w[0] -= wGradSum[0] * learningRate / samples.Length;
            w[1] -= wGradSum[1] * learningRate / samples.Length;

            b[0] -= bGradSum[0] * learningRate / samples.Length;
            b[1] -= bGradSum[1] * learningRate / samples.Length;
        }


    }

    public void Train()
    {
        network!.Weigths[0][0][0] = 0.01;
        network!.Weigths[1][0][0] = 0.01;
        TwoLayerNetwork epochVerifier = new();
        for (int i = 0; i < epochs; i++)
        {
            network!.TrainOneEpoch(samples, observed);
            epochVerifier.VerifyEpoch(samples, observed);

            bool same = network.Weigths[0][0][0] == epochVerifier.w[0];
            same &= network.Weigths[1][0][0] == epochVerifier.w[1];
            same &= network.Biases[0][0][0] == epochVerifier.b[0];
            same &= network.Biases[1][0][0] == epochVerifier.b[1];
            bool different = !same;
            if (different)
            {
                Console.WriteLine("Algorithms differ");
                break;
            }
        }
        var prediction = epochVerifier.Predict(0);
        Console.WriteLine($"0 => {prediction}");
        prediction = epochVerifier.Predict(1);
        Console.WriteLine($"1 => {prediction}");
        prediction = epochVerifier.Predict(0.5);
        Console.WriteLine($"0.5 => {prediction}");
        prediction = epochVerifier.Predict(150);
        Console.WriteLine($"150 => {prediction}");
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
        var prediction = network!.Predict([0]);
        Console.WriteLine($"0 => {prediction[0]}");
        prediction = network.Predict([1]);
        Console.WriteLine($"1 => {prediction[0]}");
        prediction = network.Predict([0.5]);
        Console.WriteLine($"0.5 => {prediction[0]}");
    }
}
