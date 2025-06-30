using System.Diagnostics;
using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class AdderTest
{
    const int epochs = 2000;
    const double learningRate = 0.025;
    double[][] samples = [];
    NeuralNetworkTrainer? network;

    public double[][] CreateTrainingData()
    {
        return samples = [[5,5], [2,2]];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(2, [1], af);
        network = networkCreator.CreateNetwork(learningRate);
    }

    internal class EpochVerifier
    {
        internal double w1=0, w2=0, b=0, diff, loss, dLoss;
        double x, y, wGrad1, wGrad2, bGrad, wGrad1Sum, wGrad2Sum, bGradSum;

        double ProcessInput(double i1, double i2)
        {
            return x = i1 * w1 + i2 * w2 + b;
        }

        double Activate(double x)
        {
            return x;
        }

        double Predict(double i1, double i2)
        {
            x = ProcessInput(i1, i2);
            return y = Activate(x);
        }
        double ActivateDerivative(double x)
        {
            return 1;
        }
        void VerifyPrediction(double i1, double i2, double observed)
        {
            y = Predict(i1, i2);
            diff = y - observed;
            loss = diff * diff;
            dLoss = 2 * diff;
            wGrad1Sum += wGrad1 = dLoss * ActivateDerivative(x) * i1;
            wGrad2Sum += wGrad2 = dLoss * ActivateDerivative(x) * i2;
            bGradSum += bGrad = dLoss * ActivateDerivative(x);

            // Console.WriteLine($"o:{observed}, i1:{i1}, w1{w1}, b{b}, XY{y}, L{loss}, dL{dLoss}, wGrd1{wGrad1}, bGrd{bGrad}, wGS1{wGrad1Sum}, bGS{bGradSum}");
        }

        internal void VerifyEpoch(double[][] samples, double[] observed)
        {
            wGrad1Sum = wGrad2Sum = bGradSum = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                VerifyPrediction(samples[i][0], samples[i][1], observed[i]);
            }
            w1 -= wGrad1Sum * learningRate / samples.Length;
            w2 -= wGrad2Sum * learningRate / samples.Length;
            b -= bGradSum* learningRate / samples.Length;
        }

    
    }

    public void Train()
    {
        EpochVerifier epochVerifier = new();
        for (int i = 0; i < epochs; i++)
        {
            double[] observed = [10, 4];
            Console.WriteLine($"Epoch: {i}");
            network!.TrainOneEpoch(samples, [[10], [4]]);
            Console.WriteLine();
            epochVerifier.VerifyEpoch(samples, observed);
            Console.WriteLine();
            bool same = network.Weigths[0][0][0] == epochVerifier.w1;
            same &= network.Weigths[0][0][1] == epochVerifier.w2;
            same &= network.Biases[0][0][0] == epochVerifier.b;
            bool different = !same;
            if (different)
            {
                Console.WriteLine("Algorithms differ");
                break;
            }
        }
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
        var prediction = network!.Predict([5, 5]);
        Console.WriteLine($"5+5 => {prediction[0]}");
        prediction = network.Predict([10,10]);
        Console.WriteLine($"10+10 => {prediction[0]}");
        prediction = network.Predict([-5, 0]);
        Console.WriteLine($"-5+0 => {prediction[0]}");
    }
}
