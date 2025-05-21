# ✈️ Flight Price Predictor

A web application that predicts flight prices based on user inputs using machine learning. Built with Streamlit, XGBoost, and robust data preprocessing pipelines.

## Live Demo

Access the deployed app here: https://flightpricepredictor.streamlit.app/

## Features

- Interactive web interface for flight price prediction
- Handles various user inputs: airline, journey date, source, destination, times, duration, stops, and additional info
- Advanced data preprocessing and feature engineering
- Trained XGBoost regression model for accurate predictions
- Real-time predictions with model and pipeline serialization

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/flight-price-predictor.git
   cd flight-price-predictor
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download or place the training data:**
   - Place your `train.csv` file in the `data/` directory.

4. **Train the preprocessor (if not already done):**
   ```sh
   python app.py
   ```
   *(The preprocessor is fitted and saved automatically on first run.)*

## Usage

1. **Start the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

2. **Open your browser and go to the provided local URL.**

3. **Enter flight details and get instant price predictions!**

## Project Structure

```
.
├── app.py                  # Main application file
├── data/
│   └── train.csv           # Training data
├── preprocessor.joblib     # Saved preprocessing pipeline
├── xgboost-model1.pkl      # Trained XGBoost model
├── requirements.txt        # Python dependencies
```

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- xgboost
- feature-engine
- joblib

*(See `requirements.txt` for full list.)*

## How it Works

- User inputs are collected via the Streamlit interface.
- Data is preprocessed using a pipeline (handling dates, times, locations, etc.).
- The processed data is fed into a trained XGBoost model.
- The predicted price is displayed instantly.

<!--
## License

This project is licensed under the MIT License.

---

**Contributions welcome!**  
Feel free to open issues or submit pull requests.
-->
