namespace EconToolbox.Desktop.ViewModels
{
    public class MainViewModel
    {
        public CapitalRecoveryViewModel CapitalRecovery { get; } = new();
        public EadViewModel Ead { get; } = new();
        public StorageCostViewModel StorageCost { get; } = new();
        public InterestDuringConstructionViewModel Idc { get; } = new();
    }
}
