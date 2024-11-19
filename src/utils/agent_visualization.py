"""
Agent interaction visualization system for IbuthoAGI.
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AgentInteractionVisualizer:
    """Visualize agent interactions and collaboration patterns."""
    
    def __init__(self):
        self.interaction_graph = nx.DiGraph()
        self.interaction_history: List[Dict[str, Any]] = []
    
    def record_interaction(
        self,
        source_agent: str,
        target_agent: str,
        interaction_type: str,
        data: Dict[str, Any]
    ) -> None:
        """Record an interaction between agents."""
        # Add to graph
        if not self.interaction_graph.has_edge(source_agent, target_agent):
            self.interaction_graph.add_edge(
                source_agent,
                target_agent,
                weight=0,
                interactions=[]
            )
        
        # Update edge weight and interactions
        self.interaction_graph[source_agent][target_agent]['weight'] += 1
        self.interaction_graph[source_agent][target_agent]['interactions'].append({
            'type': interaction_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
        
        # Record in history
        self.interaction_history.append({
            'source': source_agent,
            'target': target_agent,
            'type': interaction_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        })
    
    def create_network_visualization(self) -> go.Figure:
        """Create an interactive network visualization."""
        # Calculate layout
        pos = nx.spring_layout(self.interaction_graph)
        
        # Create edges
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for edge in self.interaction_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(
                self.interaction_graph[edge[0]][edge[1]]['weight']
            )
        
        # Create nodes
        node_x = []
        node_y = []
        node_labels = []
        node_sizes = []
        
        for node in self.interaction_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_labels.append(node)
            # Size based on number of interactions
            node_sizes.append(
                sum(self.interaction_graph[node][adj]['weight']
                    for adj in self.interaction_graph[node])
            )
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color='#888'
            ),
            hoverinfo='none'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition='bottom center',
            marker=dict(
                size=node_sizes,
                color='#1f77b4',
                line=dict(
                    width=2,
                    color='#fff'
                )
            ),
            hoverinfo='text',
            hovertext=[
                f"{label}<br>Interactions: {size}"
                for label, size in zip(node_labels, node_sizes)
            ]
        ))
        
        fig.update_layout(
            title='Agent Interaction Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def create_interaction_timeline(self) -> go.Figure:
        """Create a timeline of agent interactions."""
        fig = go.Figure()
        
        # Group interactions by type
        interaction_types = set(
            interaction['type']
            for interaction in self.interaction_history
        )
        
        for i, interaction_type in enumerate(interaction_types):
            type_interactions = [
                interaction for interaction in self.interaction_history
                if interaction['type'] == interaction_type
            ]
            
            timestamps = [
                datetime.fromisoformat(interaction['timestamp'])
                for interaction in type_interactions
            ]
            
            labels = [
                f"{interaction['source']} → {interaction['target']}"
                for interaction in type_interactions
            ]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[i] * len(timestamps),
                mode='markers+text',
                name=interaction_type,
                text=labels,
                textposition='top center',
                hoverinfo='text',
                hovertext=[
                    f"Type: {interaction_type}<br>" +
                    f"From: {interaction['source']}<br>" +
                    f"To: {interaction['target']}<br>" +
                    f"Data: {json.dumps(interaction['data'], indent=2)}"
                    for interaction in type_interactions
                ]
            ))
        
        fig.update_layout(
            title='Agent Interaction Timeline',
            yaxis=dict(
                ticktext=list(interaction_types),
                tickvals=list(range(len(interaction_types)))
            ),
            height=400 * len(interaction_types)
        )
        
        return fig
    
    def create_interaction_heatmap(self) -> go.Figure:
        """Create a heatmap of agent interactions."""
        agents = list(self.interaction_graph.nodes())
        matrix = [[0] * len(agents) for _ in range(len(agents))]
        
        # Fill interaction matrix
        for i, source in enumerate(agents):
            for j, target in enumerate(agents):
                if self.interaction_graph.has_edge(source, target):
                    matrix[i][j] = self.interaction_graph[source][target]['weight']
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=agents,
            y=agents,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title='Agent Interaction Heatmap',
            xaxis_title='Target Agent',
            yaxis_title='Source Agent'
        )
        
        return fig
    
    def create_interaction_stats(self) -> go.Figure:
        """Create statistical visualizations of interactions."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Interactions per Agent',
                'Interaction Types Distribution',
                'Interaction Trends',
                'Agent Activity Timeline'
            )
        )
        
        # Interactions per Agent
        agent_interactions = {}
        for agent in self.interaction_graph.nodes():
            agent_interactions[agent] = sum(
                self.interaction_graph[agent][adj]['weight']
                for adj in self.interaction_graph[agent]
            )
        
        fig.add_trace(
            go.Bar(
                x=list(agent_interactions.keys()),
                y=list(agent_interactions.values()),
                name='Total Interactions'
            ),
            row=1, col=1
        )
        
        # Interaction Types Distribution
        type_counts = {}
        for interaction in self.interaction_history:
            type_counts[interaction['type']] = type_counts.get(
                interaction['type'],
                0
            ) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name='Interaction Types'
            ),
            row=1, col=2
        )
        
        # Interaction Trends
        timestamps = [
            datetime.fromisoformat(interaction['timestamp'])
            for interaction in self.interaction_history
        ]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=list(range(1, len(timestamps) + 1)),
                mode='lines',
                name='Cumulative Interactions'
            ),
            row=2, col=1
        )
        
        # Agent Activity Timeline
        for agent in self.interaction_graph.nodes():
            agent_times = [
                datetime.fromisoformat(interaction['timestamp'])
                for interaction in self.interaction_history
                if interaction['source'] == agent
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=agent_times,
                    y=[agent] * len(agent_times),
                    mode='markers',
                    name=agent
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True)
        return fig
