import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple
import json
import os

# =========================
# Q NETWORK
# =========================
class QNet(nn.Module):
    def __init__(self, state_dim=7, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.net(x)


# =========================
# DQN AGENT
# =========================
class DQNAgent:
    def __init__(self, state_dim=7, action_dim=5, config=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # Q-Networks
        self.q_network = QNet(state_dim, action_dim)
        self.target_network = QNet(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        
        # RL Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.train_step = 0
        
        # History for federated learning
        self.local_updates = []
        self.global_model = None
        
        # Blockchain integration
        self.ledger = None
        
        print(f"[DQN] Initialized with state_dim={state_dim}, action_dim={action_dim}")
        
    def _build_network(self):
        return QNet(self.state_dim, self.action_dim)
    
    def act(self, state: Dict) -> Tuple[int, np.ndarray]:
        """Action selection with safety from LLM"""
        # Get observation from state dict
        obs_data = state.get('obs')
        
        # Convert to numpy if needed
        if isinstance(obs_data, list):
            obs_data = np.array(obs_data)
        elif obs_data is None:
            obs_data = np.zeros(self.state_dim)
        
        # Ensure correct dimensions
        if len(obs_data) < self.state_dim:
            obs_data = np.pad(obs_data, (0, self.state_dim - len(obs_data)))
        elif len(obs_data) > self.state_dim:
            obs_data = obs_data[:self.state_dim]
        
        obs = torch.FloatTensor(obs_data).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(obs).numpy()[0]
        
        # Apply mask if available
        mask = state.get('mask', np.ones(self.action_dim))
        if len(mask) > self.action_dim:
            mask = mask[:self.action_dim]
        elif len(mask) < self.action_dim:
            mask = np.pad(mask, (0, self.action_dim - len(mask)))
        
        if np.random.random() < self.epsilon:
            # Random action from valid actions
            valid_actions = np.where(np.array(mask) == 1)[0]
            if len(valid_actions) > 0:
                action = np.random.choice(valid_actions)
            else:
                action = np.random.randint(self.action_dim)
        else:
            # Greedy action with mask
            masked_q = q_values.copy()
            masked_q[np.array(mask) == 0] = -np.inf
            action = np.argmax(masked_q)
        
        # Log to blockchain for audit
        if self.ledger:
            self.ledger.record_action({
                'step': self.train_step,
                'action': int(action),
                'q_values': q_values.tolist(),
                'trust': state.get('trust', 0),
                'attack': state.get('attack_prob', 0),
                'timestamp': str(np.datetime64('now'))
            })
        
        return action, q_values
    
    def train(self, state, action, reward, next_state, done):
        """Train with federated learning integration"""
        # Extract obs
        state_obs = state.get('obs')
        if state_obs is None:
            state_obs = np.zeros(self.state_dim)
        elif len(state_obs) < self.state_dim:
            state_obs = np.pad(state_obs, (0, self.state_dim - len(state_obs)))
        elif len(state_obs) > self.state_dim:
            state_obs = state_obs[:self.state_dim]
            
        next_state_obs = next_state.get('obs')
        if next_state_obs is None:
            next_state_obs = np.zeros(self.state_dim)
        elif len(next_state_obs) < self.state_dim:
            next_state_obs = np.pad(next_state_obs, (0, self.state_dim - len(next_state_obs)))
        elif len(next_state_obs) > self.state_dim:
            next_state_obs = next_state_obs[:self.state_dim]
        
        # Store experience
        self.memory.append((state_obs, action, reward, next_state_obs, done))
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions)
        
        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            max_next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_step += 1
        
        # Store local update for federated learning
        self.local_updates.append({
            'step': self.train_step,
            'loss': loss.item(),
            'weights': {k: v.cpu() for k, v in self.q_network.state_dict().items()}
        })
        
        # Keep only recent updates
        if len(self.local_updates) > 100:
            self.local_updates.pop(0)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.train_step % 100 == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def get_federated_update(self) -> Dict:
        """Prepare updates for federated learning"""
        return {
            'weights': {k: v.cpu() for k, v in self.q_network.state_dict().items()},
            'steps': self.train_step,
            'epsilon': self.epsilon,
            'local_loss': np.mean([u['loss'] for u in self.local_updates[-10:]]) if self.local_updates else 0
        }
    
    def apply_federated_update(self, global_weights):
        """Apply global model update from federated learning"""
        self.q_network.load_state_dict(global_weights)
        self.target_network.load_state_dict(global_weights)
        print(f"[Federated] Applied global model update at step {self.train_step}")
    
    # =========================
    # SAVE AND LOAD METHODS - ADDED
    # =========================
    def save(self, path: str):
        """
        Save model checkpoint
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'q_state_dict': self.q_network.state_dict(),
            'target_q_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        print(f"[DQN] Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model checkpoint
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            print(f"[DQN] Warning: Model file {path} not found")
            return False
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Load state dictionaries
            self.q_network.load_state_dict(checkpoint['q_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_q_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load hyperparameters
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.train_step = checkpoint.get('train_step', 0)
            self.gamma = checkpoint.get('gamma', 0.99)
            self.epsilon_min = checkpoint.get('epsilon_min', 0.01)
            self.epsilon_decay = checkpoint.get('epsilon_decay', 0.995)
            self.batch_size = checkpoint.get('batch_size', 64)
            
            print(f"[DQN] Model loaded from {path}")
            return True
            
        except Exception as e:
            print(f"[DQN] Error loading model: {e}")
            return False
    
    def save_weights(self, path: str):
        """
        Save only weights (smaller file)
        
        Args:
            path: Path to save weights
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_network.state_dict(), path)
        print(f"[DQN] Weights saved to {path}")
    
    def load_weights(self, path: str):
        """
        Load only weights
        
        Args:
            path: Path to load weights from
        """
        if not os.path.exists(path):
            print(f"[DQN] Warning: Weights file {path} not found")
            return False
        
        try:
            self.q_network.load_state_dict(torch.load(path, map_location='cpu'))
            self.target_network.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"[DQN] Weights loaded from {path}")
            return True
        except Exception as e:
            print(f"[DQN] Error loading weights: {e}")
            return False


# =========================
# FEDERATED LEARNING
# =========================
class FederatedLearning:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.global_model = None
        self.client_updates = []
        self.aggregation_count = 0
        self.ledger = None
        
    def aggregate(self, client_updates: List[Dict]) -> Dict:
        """Aggregate weights from multiple clients using FedAvg"""
        if not client_updates:
            return None
            
        # Get first client's architecture
        first_weights = client_updates[0]['weights']
        aggregated = {}
        
        # Average weights
        for key in first_weights.keys():
            weights = [update['weights'][key].float() for update in client_updates]
            aggregated[key] = torch.mean(torch.stack(weights), dim=0)
        
        # Record aggregation to blockchain
        if self.ledger:
            self.ledger.record_aggregation({
                'round': self.aggregation_count,
                'num_clients': len(client_updates),
                'avg_loss': np.mean([u.get('local_loss', 0) for u in client_updates]),
                'timestamp': str(np.datetime64('now'))
            })
        
        self.aggregation_count += 1
        return aggregated


# =========================
# BLOCKCHAIN LEDGER
# =========================
class BlockchainLedger:
    def __init__(self, difficulty=4):
        """Initialize blockchain ledger"""
        self.chain = []
        self.current_blocks = []
        self.difficulty = difficulty
        
        # Create genesis block
        self.create_block(previous_hash='0', proof=100, data={'type': 'genesis'})
        print(f"[Blockchain] Genesis block created: {self.chain[0]['hash'][:10]}...")
    
    def create_block(self, previous_hash, proof, data):
        """Create new block"""
        import hashlib
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(np.datetime64('now')),
            'proof': proof,
            'previous_hash': previous_hash,
            'data': data,
            'hash': None
        }
        
        # Calculate hash
        block['hash'] = self.hash_block(block)
        
        self.chain.append(block)
        return block
    
    def hash_block(self, block):
        """Calculate block hash"""
        import hashlib
        import json
        block_string = json.dumps(block, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def proof_of_work(self, last_block):
        """Simple PoW"""
        last_proof = last_block['proof']
        last_hash = last_block['hash']
        proof = 0
        
        while not self.valid_proof(last_proof, proof, last_hash):
            proof += 1
            
        return proof
    
    def valid_proof(self, last_proof, proof, last_hash):
        """Validate proof of work"""
        import hashlib
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:self.difficulty] == '0' * self.difficulty
    
    def record_action(self, action_data):
        """Record an action in the blockchain"""
        self.current_blocks.append(action_data)
        
        # Create new block every 50 actions
        if len(self.current_blocks) >= 50:
            self.add_block({'actions': self.current_blocks, 'type': 'action_batch'})
            self.current_blocks = []
    
    def record_aggregation(self, agg_data):
        """Record federated learning aggregation"""
        self.add_block({'type': 'federated_aggregation', 'data': agg_data})
    
    def add_block(self, data):
        """Add block to chain"""
        previous_block = self.chain[-1]
        proof = self.proof_of_work(previous_block)
        previous_hash = previous_block['hash']
        
        self.create_block(previous_hash, proof, data)
    
    def verify_chain(self):
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current['previous_hash'] != previous['hash']:
                return False
            
            if not self.valid_proof(previous['proof'], current['proof'], previous['hash']):
                return False
                
        return True
    
    def get_history(self, limit=100):
        """Get recent history for analysis"""
        return self.chain[-limit:]