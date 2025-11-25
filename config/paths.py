"""
Path management for Trading Probability Explorer
"""
from pathlib import Path
from typing import Optional

class PathManager:
    """Manages file paths for the application"""
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_dir = self.project_root / "dataraw"
        self.sample_dir = self.data_dir / "sample"
        self.output_dir = self.project_root / "output"
        
    def get_feature_csv_path(self, filename: str) -> Path:
        """Get path to feature CSV file"""
        return self.data_dir / filename
    
    def get_trade_csv_path(self, filename: str) -> Path:
        """Get path to trade CSV file"""
        return self.data_dir / filename
    
    def get_sample_feature_csv(self) -> Path:
        """Get path to sample feature CSV"""
        return self.sample_dir / "market_features.csv"
    
    def get_output_path(self, filename: str) -> Path:
        """Get path for output file"""
        self.output_dir.mkdir(exist_ok=True)
        return self.output_dir / filename
    
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        self.data_dir.mkdir(exist_ok=True)
        self.sample_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
