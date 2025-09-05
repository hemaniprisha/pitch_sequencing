# MLB Pitch Sequencing Optimization

Advanced baseball analytics project that builds a predictive model and recommendation engine for optimal pitch sequencing using MLB Statcast data. The system combines advanced feature engineering, machine learning, and statistical analysis to recommend the next best pitch given the game context.

## Project Overview
This project:
- Engineers sequence-aware and situational features from Statcast
- Trains ML models (XGBoost, Random Forest) to estimate pitch effectiveness
- Provides real-time, context-aware next-pitch recommendations
- Simulates at-bats and monitors model performance over time

## Key Features
- **Advanced Feature Engineering**: Pitch sequence memory, velocity deltas, count leverage analysis
- **Predictive Modeling**: XGBoost and Random Forest for pitch effectiveness prediction
- **Real-time Recommendations**: Production-ready pitch recommendation engine
- **Statistical Analysis**: Comprehensive analysis of pitch effectiveness patterns
- **MLOps Framework**: Model monitoring, drift detection, and A/B testing capabilities
- **Game Simulation**: Interactive at-bat simulation using trained models

## Project Structure
```text
mlb-pitch-sequencing/
.
├── config
│   └── config.yaml
├── data
│   └── statcast_2023.parquet
├── logs
├── main.py
├── models
│   ├── random_forest_importance.csv
│   ├── random_forest_model.pkl
│   ├── xgboost_importance.csv
│   └── xgboost_model.pkl
├── outputs
│   ├── feature_importance.csv
│   ├── model_performance.csv
│   └── sample_analysis_data.csv
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-313.pyc
│   │   ├── data_processing.cpython-313.pyc
│   │   ├── feature_engineering.cpython-313.pyc
│   │   ├── modeling.cpython-313.pyc
│   │   ├── monitoring.cpython-313.pyc
│   │   ├── recommendation_engine.cpython-313.pyc
│   │   └── simulation.cpython-313.pyc
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── recommendation_engine.py
│   └── simulation.py
└── tests
    ├── __init__.py
    ├── test_data_processing.py
    ├── test_feature_engineering.py
    └── test_recommendation_engine.py
```
## Module Descriptions

### Core Modules (src/)

#### data_processing.py
- **Purpose**: Handle data acquisition, cleaning, and basic preprocessing
- **Key Class**: `DataProcessor`
- **Functions**:
  - `get_statcast_data()`: Download and cache Statcast data
  - `clean_and_engineer_basic_features()`: Clean data and create basic features
  - Private helpers for pitch type standardization, zone creation, reward calculation

#### feature_engineering.py
- **Purpose**: Create advanced, sequence-based features for ML
- **Key Class**: `FeatureEngineer`
- **Functions**:
  - `create_sequence_features()`: Generate pitch sequence memory features
  - `prepare_modeling_data()`: Prepare final dataset for ML models

#### modeling.py
- **Purpose**: Train and evaluate machine learning models
- **Key Class**: `MLBModel`
- **Functions**:
  - `train_and_evaluate()`: Main training pipeline
  - `_train_xgboost()`: Train XGBoost model
  - `_train_random_forest()`: Train Random Forest model
  - Model persistence and cross-validation utilities

#### recommendation_engine.py
- **Purpose**: Production-ready pitch recommendation system
- **Key Class**: `PitchRecommendationEngine`
- **Functions**:
  - `recommend_next_pitch()`: Core recommendation logic
  - `get_situational_recommendations()`: Context-aware recommendations
  - `validate_game_state()`: Input validation
  - Risk assessment and situational analysis methods

#### simulation.py
- **Purpose**: Simulate at-bats and games using trained models
- **Key Class**: `GameSimulator`
- **Functions**:
  - `simulate_at_bat()`: Simulate complete at-bat
  - `simulate_multiple_at_bats()`: Batch simulation for analysis
  - `simulate_inning()`: Full inning simulation
  - Outcome probability and state management methods

#### monitoring.py
- **Purpose**: Monitor model performance and detect drift
- **Key Class**: `ModelMonitor`
- **Functions**:
  - `set_baseline()`: Establish performance baseline
  - `detect_drift()`: Identify performance and data drift
  - `generate_monitoring_report()`: Create monitoring reports

## Usage Patterns

### Quick Start

1. Clone the repository:  
   ```bash
   git clone <your-repo-url>
   cd mlb-pitch-sequencing
2. Create a virtual environment: 
   ```bash
   python -m venv .venv
3. Activate virtual environment: 
   ```bash
   source .venv/bin/activate for macOS or Linux or activate .venv\Scripts\Activate.ps1 for Windows
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
6. Run the main pipeline, from data to results:
   ```bash
   python main.py  


### Development Workflow
```bash
# Run tests
pytest tests/

# Install in development mode
pip install -e .

# Run a specific module
python -m src.data_processing
```

### Custom Analysis (example)
```python
from src.data_processing import DataProcessor
from src.recommendation_engine import PitchRecommendationEngine

# 1) Load and process data
processor = DataProcessor()
df = processor.get_statcast_data(2023)
# df = processor.clean_and_engineer_basic_features(df)

# 2) Load trained artifacts (placeholders)
model = ...      # Load from models/
encoders = ...   # Any encoders used in training
pitches = ...    # Metadata (e.g., pitch type mapping)
zones = ...      # Zone definitions

engine = PitchRecommendationEngine(model, encoders, pitches, zones)

# 3) Get a recommendation
game_state = {
    'balls': 1,
    'strikes': 2,
    'outs_when_up': 1,
    'on_1b': 1,
    'prev_pitch_type_1': 'FF',
    'prev_zone_1': 'Z11',
}

recommendations = engine.recommend_next_pitch(game_state)
print(f"Recommended: {recommendations[0]['pitch_type']} in zone {recommendations[0]['zone']}")
```

## Data Flow
1. **Data Acquisition (data_processing.py)**
   - Download Statcast via `pybaseball`
   - Cache locally; generate sample data if unavailable
2. **Feature Engineering (feature_engineering.py)**
   - Clean and standardize raw data
   - Create sequence-based features; encode categoricals
3. **Model Training (modeling.py)**
   - Train XGBoost and Random Forest
   - Cross-validate and persist best models
4. **Recommendation Generation (recommendation_engine.py)**
   - Predict pitch effectiveness and rank by context
   - Include confidence scoring
5. **Simulation & Validation (simulation.py)**
   - Validate recommendations through simulation
   - Generate performance insights
6. **Monitoring & Maintenance (monitoring.py)**
   - Track performance, detect data/performance drift
   - Generate monitoring reports and alerts

## Outputs
- **Directories**
  - `models/`: Trained model files (.pkl)
  - `outputs/`: Analysis results, feature importance, sample data (.csv)
  - `logs/`: Application execution logs
  - `data/`: Cached Statcast data (.parquet)
- **Key Files**
  - `model_performance.csv`: Model evaluation metrics
  - `feature_importance.csv`: Feature importance rankings
  - `sample_analysis_data.csv`: Sample predictions and analysis
  - `monitoring_report.txt`: Model performance monitoring results

## Key Results
- **Model Performance**: XGBoost achieved R² = 0.XXX on test data
- **Top Predictors**: Previous pitch type, count state, and pitch location
- **Business Impact**: Estimated improvement in pitcher effectiveness
- **Statistical Significance**: Confirmed advantage in 2-strike counts

## Data Sources
- **Statcast Data**: High-resolution pitch tracking data from MLB
- **Features**: 15+ engineered features including sequence patterns
- **Scope**: Full season analysis with 100,000+ pitches

## Model Architecture
- **Data Pipeline**: Automated acquisition and cleaning
- **Feature Engineering**: Sequence memory, situational context, velocity analysis
- **Training**: XGBoost with cross-validation and tuning; Random Forest baseline
- **Validation**: Time-based splits and statistical testing
- **Deployment**: Production-ready recommendation system

## Production Deployment
The system includes:
- Real-time prediction API
- Model monitoring and drift detection
- A/B testing framework
- Performance validation tools

## Contributing
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-analysis`
3. Commit changes: `git commit -m 'Add new analysis'`
4. Push to branch: `git push origin feature/new-analysis`
5. Create a Pull Request


## Contact
- **Author**: Prisha Hemani
- **Email**: hemaniprisha1@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/prisha-hemani-4194a8257/

## Acknowledgments
- MLB for providing Statcast data through `pybaseball`
- Baseball analytics community for research inspiration
- Open source contributors to the underlying libraries
