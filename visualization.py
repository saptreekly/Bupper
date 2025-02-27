import streamlit as st
import numpy as np
from dataclasses import dataclass
import json
from typing import List, Dict, Tuple, Optional, Any
import time
from queue import Queue
from threading import Lock

@dataclass
class VisualizationState:
    """Track visualization state for both solvers"""
    nodes: List[Tuple[float, float]]
    edges: List[Dict]
    ant_positions: List[Tuple[float, float]] = None
    current_phase: str = "physarum"  # or "aco"

class RealTimeVisualizer:
    def __init__(self):
        self.state = None
        self.state_lock = Lock()
        self.update_queue = Queue(maxsize=100)
        self.last_update_time = 0
        self.update_interval = 0.1  # seconds between updates

    def setup_visualization(self, points: np.ndarray):
        """Initialize visualization state"""
        with self.state_lock:
            # Convert points to list for JSON serialization
            nodes = [(float(x), float(y)) for x, y in points]
            self.state = VisualizationState(
                nodes=nodes,
                edges=[],
                ant_positions=[]
            )

    def _throttled_update(self) -> bool:
        """Check if enough time has passed for a new update"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False

    def _serialize_matrix(self, matrix: Any) -> List[Dict]:
        """Convert sparse matrix or numpy array to serializable format"""
        if isinstance(matrix, dict):
            return [{"from": i, "to": j, "strength": float(val)} 
                   for (i, j), val in matrix.items()]
        else:
            rows, cols = matrix.nonzero()
            return [{"from": int(i), "to": int(j), "strength": float(matrix[i, j])}
                   for i, j in zip(rows, cols)]

    def update_physarum(self, conductivities: Dict):
        """Update Physarum visualization state"""
        if not self.state or not self._throttled_update():
            return

        with self.state_lock:
            self.state.current_phase = "physarum"
            self.state.edges = self._serialize_matrix(conductivities)
            self.state.ant_positions = []  # Clear ant positions during Physarum phase

    def update_aco(self, pheromone: np.ndarray, ant_positions: List[Tuple[int, int]]):
        """Update ACO visualization state"""
        if not self.state or not self._throttled_update():
            return

        with self.state_lock:
            self.state.current_phase = "aco"
            self.state.edges = self._serialize_matrix(pheromone)
            self.state.ant_positions = ant_positions

    def render(self):
        """Render visualization using Streamlit components"""
        if not self.state:
            return

        # Create visualization container with fixed size
        viz_container = st.empty()

        # Inject custom HTML/JavaScript for canvas rendering
        viz_html = f"""
        <div style="width: 100%; height: 500px; position: relative;">
            <canvas id="simulation_canvas" style="width: 100%; height: 100%;"></canvas>
            <div id="stats" style="position: absolute; top: 10px; right: 10px; color: white;"></div>
        </div>
        <script>
        const canvas = document.getElementById('simulation_canvas');
        const ctx = canvas.getContext('2d');
        const stats = document.getElementById('stats');

        // Animation state
        let currentState = null;
        let targetState = null;
        let transitionProgress = 0;
        const transitionDuration = 500; // ms
        let lastRenderTime = 0;
        let frameCount = 0;
        let lastFpsUpdate = 0;
        let fps = 0;

        // Setup canvas size and scaling
        function setupCanvas() {{
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }}

        // Convert data coordinates to canvas coordinates
        function dataToCanvas(x, y) {{
            const padding = 50;
            const scale = Math.min(
                (canvas.width - 2 * padding) / 100,
                (canvas.height - 2 * padding) / 100
            );
            return [
                padding + (x + 50) * scale,
                padding + (y + 50) * scale
            ];
        }}

        // Lerp between two values
        function lerp(start, end, t) {{
            return start + (end - start) * t;
        }}

        // Draw network with fluid animations
        function drawNetwork(state, t) {{
            if (!state) return;
            setupCanvas();

            // Update FPS counter
            frameCount++;
            if (Date.now() - lastFpsUpdate > 1000) {{
                fps = Math.round(frameCount * 1000 / (Date.now() - lastFpsUpdate));
                frameCount = 0;
                lastFpsUpdate = Date.now();
                stats.textContent = `FPS: ${{fps}}`;
            }}

            // Draw edges with glowing effect
            state.edges.forEach(edge => {{
                const from = state.nodes[edge.from];
                const to = state.nodes[edge.to];
                const [x1, y1] = dataToCanvas(from[0], from[1]);
                const [x2, y2] = dataToCanvas(to[0], to[1]);

                // Create gradient for glowing effect
                const gradient = ctx.createLinearGradient(x1, y1, x2, y2);

                if (state.phase === 'physarum') {{
                    // Yellow glow for Physarum
                    const alpha = Math.min(1, edge.strength);
                    gradient.addColorStop(0, `rgba(255, 255, 100, ${{alpha}})`);
                    gradient.addColorStop(0.5, `rgba(255, 255, 0, ${{alpha * 1.2}})`);
                    gradient.addColorStop(1, `rgba(255, 255, 100, ${{alpha}})`);
                }} else {{
                    // Cyan glow for ACO
                    const alpha = Math.min(1, edge.strength);
                    gradient.addColorStop(0, `rgba(100, 255, 255, ${{alpha}})`);
                    gradient.addColorStop(0.5, `rgba(0, 255, 255, ${{alpha * 1.2}})`);
                    gradient.addColorStop(1, `rgba(100, 255, 255, ${{alpha}})`);
                }}

                // Draw edge with gradient
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.strokeStyle = gradient;
                ctx.lineWidth = Math.max(2, edge.strength * 4);
                ctx.stroke();

                // Add particle effects
                if (Math.random() < 0.3 * edge.strength) {{
                    const t = Math.random();
                    const px = x1 + (x2 - x1) * t;
                    const py = y1 + (y2 - y1) * t;

                    // Particle glow
                    const particleGradient = ctx.createRadialGradient(px, py, 0, px, py, 4);
                    particleGradient.addColorStop(0, state.phase === 'physarum' ? 
                        'rgba(255, 255, 0, 0.8)' : 
                        'rgba(0, 255, 255, 0.8)');
                    particleGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

                    ctx.beginPath();
                    ctx.arc(px, py, 4, 0, 2 * Math.PI);
                    ctx.fillStyle = particleGradient;
                    ctx.fill();
                }}
            }});

            // Draw nodes with glow effects
            state.nodes.forEach((node, i) => {{
                const [x, y] = dataToCanvas(node[0], node[1]);

                // Add glow effect
                const glow = ctx.createRadialGradient(x, y, 2, x, y, 8);
                if (i === 0) {{
                    // Depot node
                    glow.addColorStop(0, 'rgba(0, 255, 0, 1)');
                    glow.addColorStop(1, 'rgba(0, 255, 0, 0)');
                }} else {{
                    glow.addColorStop(0, 'rgba(255, 255, 255, 1)');
                    glow.addColorStop(1, 'rgba(255, 255, 255, 0)');
                }}

                ctx.beginPath();
                ctx.arc(x, y, 8, 0, 2 * Math.PI);
                ctx.fillStyle = glow;
                ctx.fill();

                // Draw node
                ctx.beginPath();
                ctx.arc(x, y, i === 0 ? 6 : 4, 0, 2 * Math.PI);
                ctx.fillStyle = i === 0 ? '#00ff00' : '#ffffff';
                ctx.fill();
            }});

            // Draw ants (if in ACO phase)
            if (state.phase === 'aco' && state.ants) {{
                state.ants.forEach(ant => {{
                    const [x, y] = dataToCanvas(state.nodes[ant[0]][0], state.nodes[ant[0]][1]);

                    // Ant glow
                    const antGlow = ctx.createRadialGradient(x, y, 0, x, y, 6);
                    antGlow.addColorStop(0, 'rgba(255, 50, 50, 0.8)');
                    antGlow.addColorStop(1, 'rgba(255, 50, 50, 0)');

                    ctx.beginPath();
                    ctx.arc(x, y, 6, 0, 2 * Math.PI);
                    ctx.fillStyle = antGlow;
                    ctx.fill();

                    // Ant body
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = '#ff3232';
                    ctx.fill();
                }});
            }}
        }}

        // Animation loop with state updates
        function animate(timestamp) {{
            if (!lastRenderTime) lastRenderTime = timestamp;
            const deltaTime = timestamp - lastRenderTime;
            lastRenderTime = timestamp;

            // Get latest state
            try {{
                const stateData = JSON.parse(json.dumps({
                    "nodes": self.state.nodes,
                    "edges": self.state.edges,
                    "phase": self.state.current_phase,
                    "ants": self.state.ant_positions
                }));

                // Update state with smooth transition
                if (!currentState) {{
                    currentState = stateData;
                }} else {{
                    targetState = stateData;
                    transitionProgress = Math.min(1, transitionProgress + deltaTime / transitionDuration);

                    if (transitionProgress >= 1) {{
                        currentState = targetState;
                        targetState = null;
                        transitionProgress = 0;
                    }}
                }}

                // Draw current state
                drawNetwork(currentState, transitionProgress);
            }} catch (e) {{
                console.error('Error updating visualization:', e);
            }}

            requestAnimationFrame(animate);
        }}

        // Start animation loop
        requestAnimationFrame(animate);
        </script>
        """

        viz_container.markdown(viz_html, unsafe_allow_html=True)

# Create global visualizer instance
visualizer = RealTimeVisualizer()