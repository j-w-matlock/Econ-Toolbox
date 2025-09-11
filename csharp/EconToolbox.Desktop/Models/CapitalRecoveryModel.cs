using System;

namespace EconToolbox.Desktop.Models
{
    public static class CapitalRecoveryModel
    {
        public static double Calculate(double rate, int periods)
        {
            if (rate == 0) return 1.0 / periods;
            return rate * Math.Pow(1 + rate, periods) / (Math.Pow(1 + rate, periods) - 1);
        }
    }
}
