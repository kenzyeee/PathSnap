"""
Exploratory Data Analysis Plotting Module.

Generates:
1. Sampling interval distribution (showing ~97% 1-second sampling)
2. Trajectory point count & duration distributions across partitions
3. GPS speed and acceleration distributions
4. Road network node & arc count distributions
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any


def plot_sampling_intervals(inventory_df: pd.DataFrame, output_path: str):
    """Plots distribution of dt 1s percentages and maximum gap lengths."""
    plt.figure(figsize=(10, 4.5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(inventory_df["dt_1s_pct"], bins=20, kde=True, color="#1f77b4")
    plt.title("Distribution of 1s Sampling Compliance (%)")
    plt.xlabel("Percentage of 1s Steps per Trip (%)")
    plt.ylabel("Trip Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    sns.histplot(inventory_df["max_gap_s"], bins=20, kde=True, color="#ff7f0e")
    plt.title("Distribution of Max Time Gaps (seconds)")
    plt.xlabel("Max Gap (s)")
    plt.ylabel("Trip Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_trajectory_distributions(inventory_df: pd.DataFrame, output_path: str):
    """Plots trajectory point counts and duration by partition."""
    plt.figure(figsize=(11, 4.5))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(x="partition", y="num_points", data=inventory_df, palette="Set2")
    plt.title("GPS Observation Count per Trip by Partition")
    plt.xlabel("Partition")
    plt.ylabel("Number of Points")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x="partition", y="duration_s", data=inventory_df, palette="Set2")
    plt.title("Trip Duration (seconds) by Partition")
    plt.xlabel("Partition")
    plt.ylabel("Duration (s)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_road_network_sizes(inventory_df: pd.DataFrame, output_path: str):
    """Plots number of nodes and arcs in road networks."""
    plt.figure(figsize=(11, 4.5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(inventory_df["num_nodes"], bins=25, kde=True, color="#2ca02c")
    plt.title("Road Network Node Count Distribution")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Trip Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    sns.histplot(inventory_df["num_arcs"], bins=25, kde=True, color="#d62728")
    plt.title("Road Network Arc Count Distribution")
    plt.xlabel("Number of Arcs")
    plt.ylabel("Trip Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_speed_distributions(speeds: List[float], accelerations: List[float], output_path: str):
    """Plots speed (m/s) and acceleration (m/s^2) distributions across trajectories."""
    plt.figure(figsize=(11, 4.5))
    
    plt.subplot(1, 2, 1)
    # Clip extreme outlier GPS noise for visualization
    valid_speeds = [s for s in speeds if 0 <= s <= 50]  # up to 180 km/h
    sns.histplot(valid_speeds, bins=30, kde=True, color="#9467bd")
    plt.title("Vehicle Speed Distribution (m/s)")
    plt.xlabel("Speed (m/s)")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    valid_acc = [a for a in accelerations if -10 <= a <= 10]
    sns.histplot(valid_acc, bins=30, kde=True, color="#8c564b")
    plt.title("Vehicle Acceleration Distribution (m/s²)")
    plt.xlabel("Acceleration (m/s²)")
    plt.ylabel("Observation Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
