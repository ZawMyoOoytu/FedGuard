# utils/logger.py
import json
import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir='logs/'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Blockchain data storage
        self.blockchain_data = []
        self.blockchain = None
        
        # File paths
        self.blockchain_file = os.path.join(log_dir, 'blockchain_logs.json')
        self.summary_file = os.path.join(log_dir, 'summary.json')
        
        # Load existing data
        self._load_existing()
        
        print(f"[Logger] Initialized. Logs will be saved to: {log_dir}/")
    
    def set_blockchain(self, blockchain):
        """Connect blockchain reference to logger"""
        self.blockchain = blockchain
        print("[Logger] Blockchain connected to logger")
    
    def _load_existing(self):
        """Load existing blockchain data from file"""
        if os.path.exists(self.blockchain_file):
            try:
                with open(self.blockchain_file, 'r') as f:
                    self.blockchain_data = json.load(f)
                print(f"[Logger] Loaded {len(self.blockchain_data)} existing blockchain records")
            except:
                self.blockchain_data = []
    
    def log_blockchain(self, data):
        """Log blockchain data"""
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Add blockchain index if available
        if self.blockchain:
            data['block_index'] = len(self.blockchain.chain)
        
        # Store data
        self.blockchain_data.append(data)
        
        # Auto-save
        self._save_blockchain()
        
        return data
    
    def _save_blockchain(self):
        """Save blockchain data to file"""
        try:
            with open(self.blockchain_file, 'w') as f:
                json.dump(self.blockchain_data, f, indent=2)
        except Exception as e:
            print(f"[Logger] Error saving blockchain: {e}")
    
    def get_blockchain_data(self):
        """Return all blockchain data"""
        return self.blockchain_data
    
    def save_results(self):
        """Save final results"""
        summary = {
            'total_blocks': len(self.blockchain_data),
            'last_updated': datetime.now().isoformat(),
            'log_dir': self.log_dir
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Logger] Results saved to {self.summary_file}")
    
    def clear(self):
        """Clear all data (use with caution)"""
        self.blockchain_data = []
        self._save_blockchain()
        print("[Logger] All data cleared")