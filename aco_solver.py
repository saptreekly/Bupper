import numpy as np
from typing import List, Tuple, Dict
import random

class ACO:
    def __init__(self, 
                 n_ants: int = 20,
                 evaporation_rate: float = 0.1,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 q0: float = 0.1):
        """
        Initialize ACO solver
        
        Args:
            n_ants: Number of ants
            evaporation_rate: Pheromone evaporation rate
            alpha: Pheromone importance factor
            beta: Heuristic information importance factor
            q0: Initial pheromone value
        """
        self.n_ants = n_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0
        
    def calculate_distances(self, points: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between all points."""
        n = len(points)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i][j] = np.sqrt(np.sum((points[i] - points[j]) ** 2))
        return distances
    
    def initialize_pheromone(self, n_points: int) -> np.ndarray:
        """Initialize pheromone matrix."""
        return np.ones((n_points, n_points)) * self.q0
    
    def construct_solution(self, 
                         start: int,
                         distances: np.ndarray,
                         pheromone: np.ndarray) -> List[int]:
        """Construct a single ant's solution."""
        n = len(distances)
        unvisited = set(range(n))
        unvisited.remove(start)
        current = start
        path = [current]
        
        while unvisited:
            # Calculate probabilities
            probs = []
            for j in unvisited:
                prob = (pheromone[current][j] ** self.alpha) * \
                      ((1.0 / distances[current][j]) ** self.beta)
                probs.append((j, prob))
            
            # Select next city
            total = sum(p[1] for p in probs)
            normalized_probs = [(j, p/total) for j, p in probs]
            next_city = random.choices(
                population=[p[0] for p in normalized_probs],
                weights=[p[1] for p in normalized_probs]
            )[0]
            
            path.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        path.append(start)  # Return to start
        return path
    
    def update_pheromone(self,
                        pheromone: np.ndarray,
                        all_paths: List[List[int]],
                        all_lengths: List[float]) -> np.ndarray:
        """Update pheromone levels based on ant solutions."""
        # Evaporation
        pheromone *= (1.0 - self.evaporation_rate)
        
        # Add new pheromone
        for path, length in zip(all_paths, all_lengths):
            for i in range(len(path) - 1):
                current = path[i]
                next_city = path[i + 1]
                pheromone[current][next_city] += 1.0 / length
                pheromone[next_city][current] += 1.0 / length
                
        return pheromone
    
    def solve(self, 
             points: np.ndarray,
             n_iterations: int = 100) -> Tuple[List[int], float]:
        """
        Solve TSP using ACO
        
        Args:
            points: Array of (x, y) coordinates
            n_iterations: Number of iterations
            
        Returns:
            best_path: List of indices representing best path
            best_length: Length of best path
        """
        n_points = len(points)
        distances = self.calculate_distances(points)
        pheromone = self.initialize_pheromone(n_points)
        
        best_path = None
        best_length = float('inf')
        
        for _ in range(n_iterations):
            all_paths = []
            all_lengths = []
            
            # Construct solutions
            for _ in range(self.n_ants):
                path = self.construct_solution(0, distances, pheromone)
                length = sum(distances[path[i]][path[i+1]] 
                           for i in range(len(path)-1))
                
                all_paths.append(path)
                all_lengths.append(length)
                
                if length < best_length:
                    best_length = length
                    best_path = path.copy()
            
            # Update pheromone
            pheromone = self.update_pheromone(pheromone, all_paths, all_lengths)
            
        return best_path, best_length
