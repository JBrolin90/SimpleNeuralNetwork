using System;

namespace BackPropagation.NNLib;

public enum Operation { add, hypot }
public class Sample
{
    private double[] _sample = new double[4];
    private double observed;
    public double[] Xample => _sample;
    public double Observed => observed;
    public Sample(double a, double b, Operation op)
    {
        int i = 0;
        _sample[i++] = a;
        _sample[i++] = b;
        if (op == Operation.add)
        {
            _sample[i++] = 0;
            _sample[i++] = 1;
            observed = a + b;
        }
        else if (op == Operation.hypot)
        {
            _sample[i++] = 1;
            _sample[i++] = 0;
            observed = Math.Sqrt(a * a + b * b);
        }
    }
}
