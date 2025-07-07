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
    const int epochs = 130000;
    const double learningRate = 0.15;

    INeuralNetwork? network = null;
    NeuralNetworkTrainer? trainer;

    Normalizer normalizer = new(0, 1);

    double[][] samples = [];
    double[][] observed = [];


    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus, ActivationFunctions.Unit];
        var networkCreator = new NetworkCreator(2, [4, 4, 1], af);
        networkCreator.RandomizeWeights(-0.3, 0.3);
        network = networkCreator.CreateNetwork();
        trainer = new(network, learningRate);
    }

    void Log(double op1, double op2, double ans)
    {
        Console.WriteLine($"{op1} x {op2} = {ans}");
    }
    void Log(string s, bool newLine = true)
    {
        Console.Write(s);
        if (newLine) Console.WriteLine();

    }
    public void CreateTrainingData()
    {
        const int sampleCount = 250;
        const double from = -1;
        const double to = 1;
        const double range = to - from;
        samples = new double[sampleCount][];
        observed = new double[samples.Length][];
        double maxOp1 = 0, maxOp2 = 0;
        double minOp1 = 0, minOp2 = 0;

        samples[0] = [-1, -1]; samples[1] = [-1, 1]; samples[2] = [1, -1]; samples[3] = [1, 1];
        observed[0] = [1]; observed[1] = [-1]; observed[2] = [-1]; observed[3] = [1];
        for (int i = 4; i < sampleCount; i++)
        {
            double op1 = new Random().NextDouble() * range + from;
            double op2 = new Random().NextDouble() * range + from;
            // Log(op1, op2, op1 * op2);
            minOp1 = Math.Min(-1, op1);
            minOp2 = Math.Min(-1, op2);
            maxOp1 = Math.Max(1, op1);
            maxOp2 = Math.Max(1, op1);


            op1 = normalizer.Normalize(op1);
            op2 = normalizer.Normalize(op2);
            observed[i] = [op1 * op2];
            samples[i] = [op1, op2];
            //            Log(normalizer.Denormalize(op1), normalizer.Denormalize(op2), normalizer.Denormalize(op1 * op2));
        }
        Log($"Actual training range: [{minOp1},{minOp2}] x [{maxOp1}, {maxOp2}]");

    }
    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            double[] loss = trainer!.TrainOneEpoch(samples, observed);
            if (i % (epochs / 20) == 0)
            {
                Console.WriteLine($"Epoch {i} loss (MSE): {loss[0]}");
            }
        }
    }

    double Multiply(double a, double b)
    {
        double aa = normalizer.Normalize(a);
        double bb = normalizer.Normalize(b);
        double r = network!.Predict([aa, bb])[0];
        return normalizer.Denormalize(r);
    }

    void Multiply4Quadrants(double a, double b)
    {
        double q0 = Multiply(a, b);
        double q1 = Multiply(-a, b);
        double q2 = Multiply(a, -b);
        double q3 = Multiply(-a, -b);
        Log($"(+-{a})*(+-{b}) = [q0:{q0}, q1{q1}, q2{q2}, q3{q3}]");

    }
    public void DoIt()
    {
        CreateNetwork();
        CreateTrainingData();
        Train();


        Console.WriteLine("After training:");
        Multiply4Quadrants(0.5, 0.5);
        Multiply4Quadrants(1, 1);

        Multiply4Quadrants(0.1, 0.2);
        Multiply4Quadrants(0.2, 0.1);
        Multiply4Quadrants(0, 1);
    }
}
