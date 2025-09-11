using System;
using System.Linq;
using System.Windows.Input;
using EconToolbox.Desktop.Models;

namespace EconToolbox.Desktop.ViewModels
{
    public class EadViewModel : BaseViewModel
    {
        private string _probabilities = string.Empty;
        private string _damages = string.Empty;
        private string _result = string.Empty;

        public string Probabilities
        {
            get => _probabilities;
            set { _probabilities = value; OnPropertyChanged(); }
        }

        public string Damages
        {
            get => _damages;
            set { _damages = value; OnPropertyChanged(); }
        }

        public string Result
        {
            get => _result;
            set { _result = value; OnPropertyChanged(); }
        }

        public ICommand ComputeCommand { get; }

        public EadViewModel()
        {
            ComputeCommand = new RelayCommand(Compute);
        }

        private void Compute()
        {
            try
            {
                var p = _probabilities.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s.Trim())).ToArray();
                var d = _damages.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(s => double.Parse(s.Trim())).ToArray();
                double ead = EadModel.Compute(p, d);
                Result = $"Expected annual damage: {ead:F2}";
            }
            catch (Exception ex)
            {
                Result = $"Error: {ex.Message}";
            }
        }
    }
}
