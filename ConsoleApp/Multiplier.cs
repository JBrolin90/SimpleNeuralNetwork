using System;
using System.Security.Cryptography;
using BackPropagation.NNLib;

namespace BackPropagation;


public class Normalizer
{

    double from, to;
    double outFrom, outTo;
    double range, offset;

    public Normalizer(double from, double to, double outFrom = -1, double outTo = 1)
    {
        this.from = from;
        this.to = to;
        this.outFrom = outFrom;
        this.outTo = outTo;
        range = to - from;
        offset = 0;
    }
    public double Normalize(double x)
    {
        double n = x / range + offset;
        return n;
    }

    public double Denormalize(double x)
    {
        double d = (x - offset) * range * range;
        return d;
    }
}

public class Multiplier
{
    const int epochs = 400000;
    const double learningRate = 0.35;

    INeuralNetwork? network = null;
    NeuralNetworkTrainer? trainer;

    Normalizer normalizer = new(0, 1);

    double[][] samples = [];
    double[][] observed = [];


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(2, [4, 4, 1], af);
        networkCreator.RandomizeWeights(-1, 1);
        network = networkCreator.CreateNetwork();
        trainer = new(network, learningRate);
    }

    void Log(double op1, double op2, double ans)
    {
        Console.WriteLine($"{op1} x {op2} = {ans}");
    }
    public void CreateTrainingData()
    {
        const int sampleCount = 500;
        const double from = 0;
        const double to = 1;
        const double range = to - from;
        samples = new double[sampleCount][];
        observed = new double[samples.Length][];

        for (int i = 0; i < sampleCount; i++)
        {
            double op1 = new Random().NextDouble() * range;
            double op2 = new Random().NextDouble() * range;
            // Log(op1, op2, op1 * op2);


            op1 = normalizer.Normalize(op1);
            op2 = normalizer.Normalize(op2);
            observed[i] = [op1 * op2];
            samples[i] = [op1, op2];
//            Log(normalizer.Denormalize(op1), normalizer.Denormalize(op2), normalizer.Denormalize(op1 * op2));
        }

    }
    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = trainer!.TrainOneEpoch(samples, observed);
            if (i % (epochs / 20) == 0)
            {
                Console.WriteLine($"Epoch {i} loss: {loss[0]}");
            }
        }
    }

    double Multiply(double a, double b)
    {
        double aa = normalizer.Normalize(a);
        double bb = normalizer.Normalize(b);
        double r = network!.Predict([aa, bb])[0];
        Console.WriteLine($"{a}x{b} = {r}");
        return normalizer.Denormalize(r);
    }
    public void DoIt()
    {
        CreateNetwork();
        CreateTrainingData();
        Train();


        Console.WriteLine("After training:");
        var prediction = Multiply(0.5, 0.5);
        prediction = Multiply(1, 1);
        prediction = Multiply(0.5, 1);
        prediction = Multiply(1.5, 1);
        Multiply(2, 2);
    }
}
