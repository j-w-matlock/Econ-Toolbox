using EconToolbox.Desktop.Models;
using System.Windows.Input;

namespace EconToolbox.Desktop.ViewModels
{
    public class CapitalRecoveryViewModel : BaseViewModel
    {
        private double _rate = 5.0;
        private int _periods = 10;
        private string _result = string.Empty;

        public double Rate
        {
            get => _rate;
            set { _rate = value; OnPropertyChanged(); }
        }

        public int Periods
        {
            get => _periods;
            set { _periods = value; OnPropertyChanged(); }
        }

        public string Result
        {
            get => _result;
            set { _result = value; OnPropertyChanged(); }
        }

        public ICommand ComputeCommand { get; }

        public CapitalRecoveryViewModel()
        {
            ComputeCommand = new RelayCommand(Compute);
        }

        private void Compute()
        {
            double crf = CapitalRecoveryModel.Calculate(Rate / 100.0, Periods);
            Result = $"Capital recovery factor: {crf:F6}";
        }
    }
}
