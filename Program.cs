// See https://aka.ms/new-console-template for more information
Console.WriteLine("Hello, World!");

// declare an array of real numbers
double[] w = new double[] { 1, 3.34, -3.53, -1.22, -2.3 };
double[] b = new double[5] { 0, -1.43, 0.57, 2.0, 0.0 };
double[] x = new double[6] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

Data[] inputs = new Data[3];
inputs[0] = new Data() { input = 0.0, observed = 0.0 };
inputs[1] = new Data() { input = 0.5, observed = 1.0 };
inputs[2] = new Data() { input = 1.0, observed = 0.0 };


NeuralNetwork nn = new();
for (int i = 0; i < 1000; i++)
{
    nn.FeedForward(inputs);
}


// static double SoftPlus(double x)
// {
//     double result = Math.Log(1 + Math.Exp(x));
//     return result;
// }

// double SSR = 0.0;


// foreach (Data data in inputs)
// {
//     double input = data.input;
//     x[1] = SoftPlus(input * w[1] + b[1]);
//     x[2] = SoftPlus(input * w[2] + b[2]);

//     x[3] = x[1] * w[3];
//     x[4] = x[2] * w[4];

//     x[5] = x[3] + x[4];

//     double predicted = x[5] + b[3];
//     data.loss = predicted - data.observed;
//     data.lossSquared = Math.Pow(data.loss, 2);
//     SSR += data.lossSquared;

//     // Console.WriteLine("output = " + predicted);
//     // Console.WriteLine("Observed = " + data.observed);
//     // Console.WriteLine("loss = " + data.loss);
//     // Console.WriteLine("loss squared = " + data.lossSquared);
//     // Console.WriteLine();
//     // Console.WriteLine("===================================");
//     // Console.WriteLine();

// }
// Console.WriteLine("Sum of Squared Residuals = " + SSR);
// Console.WriteLine("===================================");
// Console.WriteLine();




