
class Data
{
    public double input;
    public double observed;
    public double loss;
    public double lossSquared;
}


class NeuralNetwork
{
    readonly bool log = true;
    public double SSR = 0.0;
    public double dSSRb3 = 0.0;
    public double dSSRw3 = 0.0;
    public double dSSRw4 = 0.0;
    public const double trainingSize = 0.1;
    #region Layers
    readonly Layer inputLayer;
    readonly Layer layer1;
    readonly Layer layer2;
    readonly Layer outputLayer;
    #endregion

    readonly Data[] inputs = [
        new Data() { input = 0.0, observed = 0.0 },
        new Data() { input = 0.5, observed = 1.0 },
        new Data() { input = 1.0, observed = 0.0 }
    ];
    public NeuralNetwork()
    {
        inputLayer = new(1, 1, Layer.UnitActivation);
        inputLayer.weights[0][0] = 1;
        inputLayer.biases[0][0] = 0;

        layer1 = new(2, 2);
        layer1.weights[0][0] = 3.34;
        layer1.weights[1][0] = -3.53;
        layer1.biases[0][0] = -1.43;
        layer1.biases[1][0] = 0.57;

        layer2 = new(2, 1, Layer.UnitActivation);
        layer2.weights[0][0] = 0.36;
        layer2.weights[0][1] = 0.63;
        layer2.biases[0][0] = 0;
        layer2.biases[0][1] = 0;

        outputLayer = new(1, 1, Layer.UnitActivation);
        outputLayer.weights[0][0] = 1;
        outputLayer.biases[0][0] = 0;

    }

    public void FeedForward(Data[] inputs)
    {
        SSR = 0.0;
        dSSRb3 = 0.0;
        foreach (var data in inputs)
        {
            double[] y0 = inputLayer.Forward([data.input]);
            double[] y1 = layer1.Forward(y0);
            double[] y2 = layer2.Forward(y1);
            double[] y3 = outputLayer.Forward(y2);

            // Log($"Input = {data.input} Output = {o3[0]}");

            data.loss = data.observed - y3[0];
            data.lossSquared = Math.Pow(data.loss, 2);
            SSR += data.lossSquared;
            dSSRb3 += -2 * data.loss;
            dSSRw3 += -2 * y1[0] * data.loss;
            dSSRw4 += -2 * y1[1] * data.loss;
            // Log($"Observ2ed = {data.observed}");
            // Log($"Loss = {data.loss}");
            // Log($"Loss Squared = {data.lossSquared}");

        }
        double stepSizeb3 = dSSRb3 * trainingSize;
        double stepSizew3 = dSSRw3 * trainingSize;
        double stepSizew4 = dSSRw4 * trainingSize;
        outputLayer.biases[0][0] -= stepSizeb3;
        layer2.weights[0][0] -= stepSizew3;
        layer2.weights[0][1] -= stepSizew4;

        Log($"w3 = {layer2.weights[0][0]} w4 = {layer2.weights[0][1]} b3 = {outputLayer.biases[0][0]}");

        // Log($"Sum of Squared Residuals = {SSR}");
        // Log($"Sum of Squared Residuals Derivative = {dSSRb3}");
        // Log($"Step Size w3 = {stepSizew3}");
        // Log($"Step Size w4 = {stepSizew4}");
        // Log($"Step Size b3 = {stepSizeb3}");
        // Log($"Bias = {outputLayer.biases[0][0]}");
        // Log("===================================");
        // Log("");
    }
    private void Log(string s)
    {
        if (log)
        {
            Logger.Log(s);
        }
    }
}

class Layer
{
    readonly bool log = false;
    public double[][] weights;
    public double[][] biases;
    public double[] outputs;
    public double[] inputs;
    public double[] deltas;
    public Func<double, double> activationFunction;

    public Layer(int inputSize, int outputSize, Func<double, double>? activationFunction = null)
    {
        activationFunction ??= SoftPlus;
        this.activationFunction = activationFunction;

        weights = new double[outputSize][];
        for (int i = 0; i < outputSize; i++)
        {
            weights[i] = new double[inputSize];
        }
        biases = new double[outputSize][];
        for (int i = 0; i < outputSize; i++)
        {
            biases[i] = new double[inputSize];
        }
        outputs = new double[outputSize];
        inputs = new double[inputSize];
        deltas = new double[outputSize];
    }

    public double[] Forward(double[] inputs)
    {
        this.inputs = inputs;

        for (int i = 0; i < outputs.Length; i++)
        {
            Log("-- Output " + i);
            double input = 0;
            for (int j = 0; j < inputs.Length; j++)
            {
                double temp = (inputs[j]) * weights[i][j] + biases[i][j];
                input += temp;
                Log($"----Input {i} temp = {temp}, input= {input}");
            }
            outputs[i] = activationFunction(input);
            Log($"--outputs[{i}] = {outputs[i]}");
        }
        return outputs;
    }
    public static double UnitActivation(double x) => x;
    public static double SoftPlus(double x)
    {
        double result = Math.Log(1 + Math.Exp(x));
        return result;
    }
    // double SoftPlusDerivative(double x)
    // {
    //     double result = 1 / (1 + Math.Exp(-x));
    //     return result;
    // }

    private void Log(string s)
    {
        if (log)
        {
            Logger.Log(s);
        }
    }


}


static class Logger
{
    public static void Log(string s)
    {
        Console.WriteLine(s);
    }
}