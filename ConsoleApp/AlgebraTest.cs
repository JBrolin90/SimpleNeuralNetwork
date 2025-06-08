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

    public void CreateTrainingData(int sampleCount = 500)
    {
        samples = new double[sampleCount][];
        observed = new double[sampleCount][];

        for (int i = 0; i < sampleCount; i++)
        {
            samples[i] = NextSample();
            observed[i] = [(samples[i][0] + samples[i][1])];
        }
    }

    public void CreateNetwork()
    {
        var networkCreator = new NetworkCreator(2, [4, 1], [ActivationFunctions.Unit, ActivationFunctions.Unit]);
        
        // Initialize weights with random values
        networkCreator.RandomizeWeights();
        
        var layerFactory = new LayerFactory();
        var nodeFactory = new NodeFactory();

        network = new NeuralNetworkTrainer(
            layerFactory, 
            nodeFactory, 
            networkCreator.Weights, 
            networkCreator.Biases, 
            networkCreator.Ys, 
            0.001,  // Learning rate
            networkCreator.ActivationFunctions);
    }

    public void Train()
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Train on all samples for this epoch
            network!.Train(samples, observed);
            
            if (epoch % 1000 == 0)
            {
                double[] testResult = network.Predict([0.1, 0.1]);
                Console.WriteLine($"Epoch {epoch}: Test(0.1+0.1) = {testResult[0]:F6}");
            }
        }
    }

    public void add(double input1, double input2)
    {
        double[] result = network!.Predict([input1, input2]);
        Console.WriteLine($"{input1} + {input2} = {result[0]:F6} (expected: {(input1 + input2):F6})");
    }

    public void DoIt()
    {
        CreateNetwork();
        
        Console.WriteLine("Testing corrected neural network framework:");
        Console.WriteLine("Before training:");
        add(0.1, 0.1);
        
        CreateTrainingData();
        Train();
        
        Console.WriteLine("After training:");
        add(0.1, 0.1);
        add(0.2, 0.3);
        add(0.05, 0.15);
        add(0.4, 0.1);
    }
}
