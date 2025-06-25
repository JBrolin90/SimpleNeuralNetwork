using System.Globalization;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class AlgebraTest
{
    const int epochs = 3000;
    Sample[] samples = [];
    readonly Random rnd = new();
    NeuralNetworkTrainer? network;

    private double normalizer = 1;

    public double RandomOperand() => rnd.NextDouble() * 2 - 1;


    public void CreateTrainingData(int sampleCount = 100)
    {
        samples = new Sample[sampleCount];
        // samples[0] = new Sample(1.0, 1.0, Operation.add);
        // return;

        for (int i = 0; i < sampleCount / 2; i++)
        {
            samples[i] = new Sample(RandomOperand() * normalizer, RandomOperand() * normalizer, Operation.add);
        }
        for (int i = sampleCount / 2; i < sampleCount; i++)
        {
            samples[i] = new Sample(RandomOperand() * normalizer, RandomOperand() * normalizer, Operation.add);
        }
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(4, [2, 2, 1], af);

        // Initialize weights with random values
        networkCreator.RandomizeWeights(-1, 1);
        network = networkCreator.CreateNetwork(0.1);

    }

    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = network!.TrainOneEpoch(samples);
            Console.WriteLine($"Loss = {loss[0]}");
        }
    }

    public void Add(double input1, double input2)
    {
        double[] result = network!.Predict([input1, input2, 0, 1]);
        Console.WriteLine($"{input1} + {input2} = {result[0]:F6} (expected: {(input1 + input2):F6})");
    }
    public double Hypot(double input1, double input2)
    {
        double[] result = network!.Predict([input1, input2, 1, 0]);
        Console.WriteLine($"sqrt({input1}^2 + {input2}^2) = {result[0]:F6} (expected: {Math.Sqrt(input1 * input1 + input2 * input2):F6})");
        return result[0];
    }

    public void DoIt()
    {
        CreateNetwork();

        Console.WriteLine("Before training:");

        CreateTrainingData();
        double[][][] Weights = network!.Weigths;
        for (int i = 0; i < Weights.Length; i++)
        {
            double[][] nodes = Weights[i];
            for (int j = 0; j < nodes.Length; j++)
            {
                double[] inputs = nodes[j];
                for (int k = 0; k < inputs.Length; k++)
                {
                    inputs[k] = 0.3;
                }
            }
        }

        Train();

        Console.WriteLine("After training:");
        Add(0.1, 0.1);
        Add(0.2, 0.3);
        Add(0.05, 0.15);
        Add(0.4, 0.1);
        Hypot(0.1, 0.1);
    }
}
