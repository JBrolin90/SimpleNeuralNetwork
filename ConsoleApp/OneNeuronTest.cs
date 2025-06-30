using System.Diagnostics;
using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class OneNeuronTest
{
    const int epochs = 500;
    double[] samples = [];
    NeuralNetworkTrainer? network;

    public double[] CreateTrainingData()
    {
        return samples = [0, 1];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [1], af);
        //networkCreator.RandomizeWeights(-2, 2);
        network = networkCreator.CreateNetwork(0.1);
    }

    internal class EpochVerifier
    {
        double w=0, b=0, diff, loss, dLoss;
        double x, y, wGrad, bGrad, wGradSum, bGradSum;

        double ProcessInput(double i)
        {
            return x = i * w + b;
        }

        double Activate(double x)
        {
            return x;
        }

        double Predict(double i)
        {
            x = ProcessInput(i);
            return y = Activate(x);
        }
        double ActivateDerivative(double x)
        {
            return 1;
        }
        void VerifyPrediction(double i, double observed)
        {
            y = Predict(i);
            diff = y - observed;
            loss = diff * diff;
            dLoss = 2 * diff;
            wGradSum += wGrad = dLoss * ActivateDerivative(x) * i;
            bGradSum += bGrad = dLoss * ActivateDerivative(x);

            Console.WriteLine($"o:{observed}, i:{i}, w{w}, b{b}, XY{y}, L{loss}, dL{dLoss}, wGrd{wGrad}, bGrd{bGrad}, wGS{wGradSum}, bGS{bGradSum}");
        }

        internal void VerifyEpoch(double[] samples, double[] observed)
        {
            wGradSum = bGradSum = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                VerifyPrediction(samples[i], observed[i]);
            }
            w -= wGradSum*0.1 / samples.Length;
            b -= bGradSum*0.1 / samples.Length;
        }

    
    }

    public void Train()
    {
        EpochVerifier epochVerifier = new();
        for (int i = 0; i < epochs; i++)
        {
            double[] observed = [0.3, 0.6];
            Console.WriteLine($"Epoch: {i}");
            network!.TrainOneEpoch( [ [0], [1] ] , [[0.3], [0.6]]);
            Console.WriteLine();
            epochVerifier.VerifyEpoch(samples, observed);
            Console.WriteLine();
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
        var prediction = network!.Predict([0]);
        Console.WriteLine($"0 => {prediction[0]}");
        prediction = network.Predict([10]);
        Console.WriteLine($"10 => {prediction[0]}");
        prediction = network.Predict([-5]);
        Console.WriteLine($"-5 => {prediction[0]}");
    }
}
