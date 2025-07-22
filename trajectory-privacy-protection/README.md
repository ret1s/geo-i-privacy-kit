# Trajectory Privacy Protection

This project implements an algorithm to protect trajectory privacy in location-based services by generating realistic, privacy-preserving fake trajectories. The implementation incorporates geo-indistinguishability, quality of service, realism, start and end point protection, and continuous reporting.

## Project Structure

```
trajectory-privacy-protection
├── src
│   ├── __init__.py
│   ├── main.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── privacy_engine.py
│   │   ├── geo_indistinguishability.py
│   │   └── trajectory_generator.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── osm_data_manager.py
│   │   ├── coordinate_transformer.py
│   │   └── network_snapper.py
│   ├── algorithms
│   │   ├── __init__.py
│   │   ├── realism_checker.py
│   │   ├── quality_of_service.py
│   │   └── point_protection.py
│   ├── visualization
│   │   ├── __init__.py
│   │   ├── map_generator.py
│   │   └── trajectory_plotter.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── constants.py
│   └── utils
│       ├── __init__.py
│       ├── geometry_helpers.py
│       └── validation.py
├── tests
│   ├── __init__.py
│   ├── test_privacy_engine.py
│   ├── test_trajectory_generator.py
│   └── test_realism_checker.py
├── data
│   └── sample_trajectories.json
├── output
│   └── .gitkeep
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd trajectory-privacy-protection
pip install -r requirements.txt
```

## Usage

To run the trajectory privacy protection algorithm, execute the following command:

```bash
python src/main.py
```

This will initialize the algorithm and generate privacy-preserving fake trajectories based on the specified parameters.

## Libraries Used

- **Folium**: For generating interactive maps.
- **GeoPandas**: For geospatial data processing.
- **Shapely**: For geometric operations.

## Testing

Unit tests are provided to ensure the functionality of the core components. To run the tests, use:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.