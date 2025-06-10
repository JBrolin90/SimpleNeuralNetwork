using System.Security.AccessControl;
using BackPropagation.NNLib;

namespace BackPropagation;

public class AlgebraTest
{
    const int epochs = 10000;
    double[][] samples = [];
    double[][] observed = [];
    readonly Random rnd = new();
    NeuralNetworkTrainer? network;

    public double[] NextSample() => [rnd.NextDouble() / 2, rnd.NextDouble() / 2];

    public void CreateTrainingData(int sampleCount = 1000)
    {
        samples = new double[sampleCount][];
        observed = new double[sampleCount][];

        for (int i = 0; i < sampleCount/2; i++)
        {
            samples[i] = NextSample(); samples[i] = samples[i].Append(0).ToArray();
            observed[i] = [(samples[i][0] + samples[i][1])];
        }
        for (int i = sampleCount/2; i < sampleCount; i++)
        {
            samples[i] = NextSample(); samples[i] = samples[i].Append(1).ToArray();
            observed[i] = [(Math.Sqrt(samples[i][0]* samples[i][0] + samples[i][1]* samples[i][1]))];
        }
    }

    public void CreateTrainingDataSeparate(string functionType, int sampleCount = 1000)
    {
        samples = new double[sampleCount][];
        observed = new double[sampleCount][];

        if (functionType == "addition")
        {
            for (int i = 0; i < sampleCount; i++)
            {
                double[] baseSample = NextSample();
                samples[i] = [baseSample[0], baseSample[1], 0];
                observed[i] = [(samples[i][0] + samples[i][1])];
            }
        }
        else if (functionType == "hypot")
        {
            for (int i = 0; i < sampleCount; i++)
            {
                double[] baseSample = NextSample();
                samples[i] = [baseSample[0], baseSample[1], 1];
                observed[i] = [Math.Sqrt(samples[i][0] * samples[i][0] + samples[i][1] * samples[i][1])];
            }
        }
    }

    public void TrainSequentially()
    {
        // First train on addition
        Console.WriteLine("Training addition function...");
        CreateTrainingDataSeparate("addition", 2000);

        for (int epoch = 0; epoch < epochs / 2; epoch++)
        {
            double[] SSR = network!.Train(samples, observed);

            if (epoch % 500 == 0)
            {
                double[] testResult = network.Predict([0.1, 0.1, 0]);
                Console.WriteLine($"Addition Epoch {epoch}: Test(0.1+0.1) = {testResult[0]:F6}");
            }
        }

        // Then train on hypot
        Console.WriteLine("Training hypot function...");
        CreateTrainingDataSeparate("hypot", 2000);
        for (int epoch = 0; epoch < epochs / 2; epoch++)
        {
            double[] SSR = network!.Train(samples, observed);

            if (epoch % 500 == 0)
            {
                double[] testResult = network.Predict([0.1, 0.1, 1]);
                Console.WriteLine($"Hypot Epoch {epoch}: Test hypot(0.1,0.1) = {testResult[0]:F6}");
            }
        }
    }

    public void CreateTrainingDataMixed(int sampleCount = 2000)
    {
        samples = new double[sampleCount][];
        observed = new double[sampleCount][];

        for (int i = 0; i < sampleCount; i++)
        {
            double[] baseSample = NextSample();

            // Alternate between functions for balanced training
            if (i % 2 == 0)
            {
                // Addition samples
                samples[i] = [baseSample[0], baseSample[1], 0];
                observed[i] = [(samples[i][0] + samples[i][1])];
            }
            else
            {
                // Hypot samples  
                samples[i] = [baseSample[0], baseSample[1], 1];
                observed[i] = [Math.Sqrt(samples[i][0] * samples[i][0] + samples[i][1] * samples[i][1])];
            }
        }
    }

    public void TrainMixed()
    {
        CreateTrainingDataMixed(4000);  // More training data

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double[] SSR = network!.Train(samples, observed);

            if (epoch % 1000 == 0)
            {
                double[] addResult = network.Predict([0.1, 0.1, 0]);
                double[] hypotResult = network.Predict([0.1, 0.1, 1]);
                double expectedHypot = Math.Sqrt(0.1 * 0.1 + 0.1 * 0.1);

                Console.WriteLine($"Epoch {epoch}: Add(0.1+0.1) = {addResult[0]:F6} (exp: 0.200000)");
                Console.WriteLine($"              Hypot(0.1,0.1) = {hypotResult[0]:F6} (exp: {expectedHypot:F6})");
            }
        }
    }

    public void TrainWithCurriculum()
    {
        // Phase 1: Train only on addition (easier function)
        Console.WriteLine("Phase 1: Learning addition...");
        CreateTrainingDataSeparate("addition", 1000);

        for (int epoch = 0; epoch < 2000; epoch++)
        {
            network!.Train(samples, observed);

            if (epoch % 500 == 0)
            {
                double[] testResult = network.Predict([0.1, 0.1, 0]);
                Console.WriteLine($"Addition Phase Epoch {epoch}: {testResult[0]:F6}");
            }
        }

        // Phase 2: Mixed training with both functions
        Console.WriteLine("Phase 2: Learning both functions...");
        CreateTrainingDataMixed(3000);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            network!.Train(samples, observed);

            if (epoch % 1000 == 0)
            {
                double[] addResult = network.Predict([0.1, 0.1, 0]);
                double[] hypotResult = network.Predict([0.1, 0.1, 1]);
                double expectedHypot = Math.Sqrt(0.1 * 0.1 + 0.1 * 0.1);

                Console.WriteLine($"Mixed Phase Epoch {epoch}:");
                Console.WriteLine($"  Add(0.1+0.1) = {addResult[0]:F6} (exp: 0.200000)");
                Console.WriteLine($"  Hypot(0.1,0.1) = {hypotResult[0]:F6} (exp: {expectedHypot:F6})");
            }
        }
    }

    public void CreateBetterNetwork()
    {
        // Try different architectures based on complexity
        Func<double, double>[] af = [ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.Unit];

        // Deeper network: 3 -> 12 -> 8 -> 4 -> 1
        var networkCreator = new NetworkCreator(3, [12, 8, 4, 1], af);

        // Smaller weight initialization for better stability
        networkCreator.RandomizeWeights(-0.2, 0.2);

        var layerFactory = new LayerFactory();
        var nodeFactory = new NodeFactory();

        network = new NeuralNetworkTrainer(
            layerFactory,
            nodeFactory,
            networkCreator.Weights,
            networkCreator.Biases,
            networkCreator.Ys,
            0.01,  // Higher learning rate
            networkCreator.ActivationFunctions);
    }

    public void CreateNetwork()
    {
        Func<double, double>[] af = [ActivationFunctions.ReLU, ActivationFunctions.ReLU, ActivationFunctions.Unit];
        //       Func<double, double>[] af = [ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus, ActivationFunctions.SoftPlus];
        var networkCreator = new NetworkCreator(3, [8, 6, 1], af);

        // Initialize weights with random values
        networkCreator.RandomizeWeights(-0.5, 0.5);

        var layerFactory = new LayerFactory();
        var nodeFactory = new NodeFactory();

        network = new NeuralNetworkTrainer(
            layerFactory,
            nodeFactory,
            networkCreator.Weights,
            networkCreator.Biases,
            networkCreator.Ys,
            0.005,  // Learning rate
            networkCreator.ActivationFunctions);
    }

    public void Train()
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Train on all samples for this epoch
            double[] SSR = network!.Train(samples, observed);
            
            if (epoch % 1000 == 0)
            {
                double[] testResult = network.Predict([0.1, 0.1, 0]);
                Console.WriteLine($"Epoch {epoch}: Test(0.1+0.1) = {testResult[0]:F6}  SSR={SSR[0]}");
            }
        }
    }

    public void Add(double input1, double input2)
    {
        double[] result = network!.Predict([input1, input2, 0]);
        Console.WriteLine($"{input1} + {input2} = {result[0]:F6} (expected: {(input1 + input2):F6})");
    }
    public double Hypot(double input1, double input2)
    {
        double[] result = network!.Predict([input1, input2, 1]);
        Console.WriteLine($"sqrt({input1}^2 + {input2}^2) = {result[0]:F6} (expected: {Math.Sqrt(input1 * input1 + input2 * input2):F6})");
        return result[0];
    }

    public void DoIt()
    {
        //CreateNetwork();
        CreateBetterNetwork();

        Console.WriteLine("Testing corrected neural network framework:");
        Console.WriteLine("Before training:");
        Add(0.1, 0.1);

        //CreateTrainingData();
        //Train();
        //TrainSequentially();
        //TrainMixed();
        TrainWithCurriculum();

        Console.WriteLine("After training:");
        Add(0.1, 0.1);
        Add(0.2, 0.3);
        Add(0.05, 0.15);
        Add(0.4, 0.1);
        Hypot(0.1, 0.1);
    }
}
