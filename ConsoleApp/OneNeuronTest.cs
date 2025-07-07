using System.Diagnostics;
using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class OneNeuronTest
{
    const int epochs = 500;
    double[] samples = [];

    INeuralNetwork? network;
    NeuralNetworkTrainer? trainer;
    public bool EnableVerboseOutput { get; set; } = true;

    public INeuralNetwork GetNetwork()
    {
        if (network == null)
            throw new InvalidOperationException("Network has not been created. Call CreateNetwork() first.");
        return network;
    }

    public double[] CreateTrainingData()
    {
        return samples = [0, 1];
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(1, [1], af);
        //networkCreator.RandomizeWeights(-2, 2);
        network = networkCreator.CreateNetwork();
        trainer = new(network, 0.1);
    }

    internal class EpochVerifier
    {
        double w=0, b=0, diff, loss, dLoss;
        double x, y, wGrad, bGrad, wGradSum, bGradSum;
        private readonly bool enableVerboseOutput;

        public EpochVerifier(bool enableVerboseOutput = true)
        {
            this.enableVerboseOutput = enableVerboseOutput;
        }

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

            if (enableVerboseOutput)
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
        EpochVerifier epochVerifier = new(EnableVerboseOutput);
        for (int i = 0; i < epochs; i++)
        {
            double[] observed = [0.3, 0.6];
            if (EnableVerboseOutput)
                Console.WriteLine($"Epoch: {i}");
            trainer!.TrainOneEpoch( [ [0], [1] ] , [[0.3], [0.6]]);
            if (EnableVerboseOutput)
                Console.WriteLine();
            epochVerifier.VerifyEpoch(samples, observed);
            if (EnableVerboseOutput)
                Console.WriteLine();
        }
    }


    private double[][][] FixedWeights()
    {
        double FixedWeight(double _) => 0;
        NetworkCreator.ApplyOn3dArr(network!.Weigths, FixedWeight);
        return network.Weigths;
    }
    public void Execute()
    {
        DoIt();
    }

    public void DoIt()
    {
        CreateNetwork();

        CreateTrainingData();

        Train();

        if (EnableVerboseOutput)
        {
            Console.WriteLine("After training:");
            var prediction = network!.Predict([0]);
            Console.WriteLine($"0 => {prediction[0]}");
            prediction = network.Predict([10]);
            Console.WriteLine($"10 => {prediction[0]}");
            prediction = network.Predict([-5]);
            Console.WriteLine($"-5 => {prediction[0]}");
        }
    }
}
