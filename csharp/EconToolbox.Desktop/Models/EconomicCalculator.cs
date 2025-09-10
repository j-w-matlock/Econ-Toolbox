using System;
using System.Linq;
using System.Collections.Generic;

namespace EconToolbox.Desktop.Models
{
    public static class EconomicCalculator
    {
        public static double CapitalRecoveryFactor(double rate, int periods)
        {
            if (rate == 0) return 1.0 / periods;
            return rate * Math.Pow(1 + rate, periods) / (Math.Pow(1 + rate, periods) - 1);
        }

        public static double EadTrapezoidal(IEnumerable<double> probabilities, IEnumerable<double> damages)
        {
            var p = probabilities.ToArray();
            var d = damages.ToArray();
            if (p.Length != d.Length) throw new ArgumentException("Probability and damage counts must match");
            double sum = 0;
            for (int i = 0; i < p.Length - 1; i++)
            {
                sum += 0.5 * (d[i] + d[i + 1]) * (p[i] - p[i + 1]);
            }
            return sum;
        }

        public static double UpdatedStorageCost(double tc, double sp, double storageReallocated, double totalUsableStorage)
        {
            return (tc - sp) * storageReallocated / totalUsableStorage;
        }

        public static double InterestDuringConstruction(double totalInitialCost, double rate, int months, double[]? costs = null, string[]? timings = null)
        {
            if (months <= 0) return 0.0;
            double monthlyRate = rate / 12.0;
            if (costs == null)
            {
                double monthlyCost = totalInitialCost / months;
                costs = Enumerable.Repeat(monthlyCost, months).ToArray();
                timings = new string[months];
                timings[0] = "beginning";
                for (int i = 1; i < months; i++) timings[i] = "middle";
            }
            else if (timings == null)
            {
                timings = Enumerable.Repeat("middle", costs.Length).ToArray();
            }
            double idc = 0.0;
            for (int i = 0; i < costs.Length; i++)
            {
                string timing = timings![i];
                double remaining;
                if (timing == "beginning") remaining = months - i;
                else if (timing == "end") remaining = months - i - 1;
                else remaining = months - i - 0.5;
                idc += costs[i] * monthlyRate * remaining;
            }
            return idc;
        }
    }
}
