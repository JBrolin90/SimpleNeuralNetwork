using System.Globalization;
using System.Reflection.Metadata.Ecma335;
using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class AlgebraTest
{
    const int epochs = 500000;
    Sample[] samples = [];
    readonly Random rnd = new();
    NeuralNetworkTrainer? network;

    private double normalizer = 1;

    public double RandomOperand() => (rnd.NextDouble() * 10 - 5);


    public void CreateTrainingData(int sampleCount = 60)
    {
        samples = new Sample[sampleCount];

        for (int i = 0; i < sampleCount / 2; i++)
        {
            samples[i] = new Sample(RandomOperand(), RandomOperand(), Operation.add, normalizer);
        }
        for (int i = sampleCount / 2; i < sampleCount; i++)
        {
            samples[i] = new Sample(RandomOperand(), RandomOperand(), Operation.hypot, normalizer);
        }
    }


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(4, [8, 16, 1], af);

        // Initialize weights with random values
        networkCreator.RandomizeWeights(-1, 1);
        network = networkCreator.CreateNetwork(0.001);
    }

    public double Train()
    {
        double[] loss = new double[1];
        for (int i = 0; i < epochs; i++)
        {
            loss = network!.TrainOneEpoch(samples);
            if (i % (epochs / 10) == 0)
            {
                Console.WriteLine($"Epoch {i} => Loss = {loss[0]}");
            }
        }
        Console.WriteLine($"Loss = {loss[0]}");
        return loss[0];
    }

    public void Add(double input1, double input2)
    {
        double[] result = network!.Predict([input1/normalizer, input2/normalizer, 0, 1]);
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

        Train();

        Console.WriteLine("After training:");
        Add(1, 1);
        Add(2, 3);
        Add(0.5, 1.5);
        Add(4, 1);
        Hypot(3, 4);
    }
}
