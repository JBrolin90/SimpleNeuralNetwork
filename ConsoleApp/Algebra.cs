using BackPropagation.NNLib;

namespace BackPropagation;

public class Algebra
{
    const int epochs = 10000;
    double[][] samples = [];
    double[][] observed = [];
    readonly Random rnd = new();
    SimpleAdder? network;

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

    public void Train()
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0;
            
            for (int i = 0; i < samples.Length; i++)
            {
                double prediction = network!.Forward(samples[i]);
                double error = observed[i][0] - prediction;
                totalError += error * error;
                network.Backward(error);
            }
            
            if (epoch % 1000 == 0)
            {
                double avgError = totalError / samples.Length;
                double testPred = network!.Forward([0.1, 0.1]);
                Console.WriteLine($"Epoch {epoch}: Avg Error = {avgError:F6}, Test(0.1+0.1) = {testPred:F6}");
            }
        }
    }

    public void DoIt()
    {
        network = new SimpleAdder();
        
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

    public double add(double a, double b)
    {
        double result = network!.Forward([a, b]);
        Console.WriteLine($"{a} + {b} = {result:F6} (expected: {a + b:F6})");
        return result;
    }
}

// Simple neural network specifically for addition
public class SimpleAdder
{
    double[] w1 = new double[2]; // Input to hidden weights
    double[] w2 = new double[2]; // Hidden to output weights  
    double b1 = 0; // Hidden bias
    double b2 = 0; // Output bias
    double learningRate = 0.01;
    
    // Store values for backprop
    double[] inputs = new double[2];
    double hidden;
    double output;
    
    public SimpleAdder()
    {
        Random rnd = new Random(42);
        // Initialize small random weights
        for (int i = 0; i < 2; i++)
        {
            w1[i] = (rnd.NextDouble() - 0.5) * 0.1;
            w2[i] = (rnd.NextDouble() - 0.5) * 0.1;
        }
        b1 = (rnd.NextDouble() - 0.5) * 0.1;
        b2 = (rnd.NextDouble() - 0.5) * 0.1;
    }
    
    public double Forward(double[] input)
    {
        inputs[0] = input[0];
        inputs[1] = input[1];
        
        // Hidden layer (just sum the inputs)
        hidden = inputs[0] * w1[0] + inputs[1] * w1[1] + b1;
        
        // Output layer
        output = hidden * w2[0] + inputs[0] * w2[1] + b2; // Direct connection too
        
        return output;
    }
    
    public void Backward(double error)
    {
        // Update output layer weights
        w2[0] += learningRate * error * hidden;
        w2[1] += learningRate * error * inputs[0];
        b2 += learningRate * error;
        
        // Update hidden layer weights
        double hiddenError = error * w2[0];
        w1[0] += learningRate * hiddenError * inputs[0];
        w1[1] += learningRate * hiddenError * inputs[1];
        b1 += learningRate * hiddenError;
    }
}
