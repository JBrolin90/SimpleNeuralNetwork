using BackPropagation.NNLib;

namespace BackPropagation;

public class Algebra
{
    const int epochs = 50000;
    Func<double, double>[] activationFunctions = [
        ActivationFunctions.SoftPlus,
        ActivationFunctions.SoftPlus,
        ActivationFunctions.SoftPlus,
        ActivationFunctions.SoftPlus
    ];

    double[][] samples = [];
    double[][] observed = [];
    readonly Random rnd = new();
    NeuralNetworkTrainer? nn2;


    public double[] NextSample() => [rnd.NextDouble() / 2, rnd.NextDouble() / 2];

    public void CreateTrainingData(int sampleCount = 3)
    {
        samples = new double[sampleCount][];
        observed = new double[sampleCount][];

        for (int i = 0; i < sampleCount; i++)
        {
            samples[i] = NextSample();
            observed[i] = [(samples[i][0] + samples[i][1])];
        }
    }

    public void Train()
    {
        for (int i = 0; i < epochs; i++)
        {
            nn2.Train(samples, observed);
        }
    }

    public void DoIt()
    {
        double[] inp = [0.1, 0.1];

        NetworkCreator creator = new(2, [2, 4, 4, 1], activationFunctions);
        creator.RandomizeWeights();
        nn2 = creator.CreateNetwork();
        add(0.1,0.1);
        CreateTrainingData();
        Train();
        add(0.1, 0.1);

        double[] result = nn2.Predict(inp);

    }

    public double add(double a, double b)
    {
        // a /= 10; b /= 10;
        double result = nn2.Predict([a, b])[0];
        Console.WriteLine($"{a} + {b} = {result}");
        return result;
    }
}
