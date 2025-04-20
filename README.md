<div align="center">

# Geo-Indistinguishability Privacy Kit for Trajectory Protection

</div>

This repository contains the implementation of a privacy-preserving approach to trajectory data protection, developed as part of a Master's Thesis in Computer Science. The system employs differential privacy concepts, specifically geo-indistinguishability, to protect users' location data while maintaining utility for spatial analysis.

The framework generates realistic yet privacy-preserving synthetic trajectories that follow road networks and preserve important characteristics of the original paths, all while providing formal privacy guarantees.

## Algorithm Overview

The current algorithm employs a multi-stage process to generate privacy-protected synthetic trajectories:

### 1. Road Network Acquisition
- Download road network data for a specified geographical location using OpenStreetMap.

### 2. Real Trajectory Generation
- Create a complex real trajectory following the road network using a realistic movement model with strategic destination selection.

### 3. Privacy Mechanism Application
- Apply geo-indistinguishability noise to the starting point using a planar Laplace mechanism.
- Generate multiple synthetic trajectories that balance privacy and utility.
- Ensure endpoints are within 1-2 road segments of real endpoints (but not identical).
- Create deliberate diversions in the middle portions of trajectories.

### 4. Visualization
- Generate comparative visualizations of real vs. synthetic trajectories on road networks.

## Technologies Used

The implementation utilizes the following technologies:

- **Python**: Core programming language
- **NetworkX**: Graph theory library for road network representation and pathfinding
- **OSMnx**: OpenStreetMap data acquisition and processing
- **GeoPandas**: Spatial data operations and analysis
- **Matplotlib & Contextily**: Visualization and mapping
- **NumPy**: Mathematical operations and probability distributions

## Mathematical Background

The privacy mechanism is built on the following mathematical foundations:

### Geo-Indistinguishability
Geo-indistinguishability extends differential privacy to the spatial domain. For two locations \(x\) and \(y\), the privacy mechanism \(K\) satisfies ε-geo-indistinguishability if:

$$
\frac{\Pr[K(x) \in S]}{\Pr[K(y) \in S]} \le e^{\varepsilon d(x,y)}
$$

Where:
- d(x,y): the distance between locations x and y
- ε (epsilon): the privacy parameter (lower values provide stronger privacy)
- S: any subset of possible outputs

### Planar Laplace Mechanism
The implementation uses a planar Laplace distribution to add calibrated noise to locations:

$$
D_{\varepsilon}(x) \propto e^{-\varepsilon||x||}
$$

Where \(||x||\) is the Euclidean distance from the origin.

The noise generation process is as follows:
- Sample angle θ uniformly from the interval [0, 2π).
- Sample radius r from an exponential distribution with parameter ε.
- Convert polar coordinates (r, θ) to Cartesian coordinates: $(x, y) = (r \cdot \cos(\theta), r \cdot \sin(\theta))$

### Similarity-Based Synthetic Trajectory Generation
The synthetic trajectories are generated using a biased random walk on the road network graph, where:
- The bias is controlled by a similarity parameter (0.0 to 1.0)
- The probability of following the real path decreases with distance from endpoints
- Strategic diversions are introduced for better privacy protection
- Graph-based constraints ensure trajectories follow real-world road structures

This approach provides a strong balance between privacy protection and maintaining trajectory utility.

## Usage

The code can be executed to generate multiple privacy-preserving synthetic trajectories for a given real location. Parameters such as (ε) (privacy level), similarity factor, and trajectory complexity can be adjusted to meet specific requirements.

## Example

#### Real vs. Synthetic Trajectories
![Example 1](example/1.png)
![Example 2](example/2.png)
<!-- ![Example 3](example/3.png) -->
![Example 4](example/4.png)
![Example 5](example/5.png)

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/ret1s/geo-i-privacy-kit.git
    ```
2. Navigate to the `geo-i-privacy-kit` directory:
    ```bash
    cd geo-i-privacy-kit
    ```
3. Follow the instructions in the `README.md` file within the `geo-i-privacy-kit` directory to set up dependencies and execute the scripts.

## Requirements

- Python 3.x
- Required libraries are listed in `requirements.txt`. Install them using:
  ```bash
  pip install -r requirements.txt
  ```

## Contact

For any questions or feedback, feel free to reach out at [ngmhah@gmail.com].