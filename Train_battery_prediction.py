# train_battery_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import os


class BatteryPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def load_and_preprocess_data(self, csv_file):
        """Load and preprocess battery data"""
        # Check if file exists
        if not os.path.exists(csv_file):
            print("ERROR: File '" + csv_file + "' not found!")
            print("Current working directory: " + os.getcwd())
            return None, None

        df = pd.read_csv(csv_file)
        print("Loaded " + str(len(df)) + " samples")
        print("Data sample:")
        print(df.head())

        # Check if we have enough data
        if len(df) < 10:
            print("Not enough data for training. Need at least 10 samples.")
            return None, None

        # Features: current battery, current speed, current drain rate
        # Target: next battery level (we'll shift the data)
        features = []
        targets = []

        # Create sequences for prediction
        for i in range(len(df) - 1):
            current_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            features.append([
                current_row['battery_level'],
                current_row['motor_speed'],
                current_row['battery_drain_rate']
            ])

            targets.append(next_row['battery_level'])

        X = np.array(features)
        y = np.array(targets)

        print("Created " + str(len(X)) + " training samples")
        return X, y

    def create_model(self, input_shape):
        """Create neural network for battery prediction"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear')  # Predict battery level
        ])

        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, csv_file):
        """Train the battery prediction model"""
        # Load data
        X, y = self.load_and_preprocess_data(csv_file)

        if X is None or y is None:
            print("Failed to load data. Cannot train model.")
            return None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("Training samples: " + str(X_train.shape[0]))
        print("Test samples: " + str(X_test.shape[0]))

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train model
        self.model = self.create_model((X_train.shape[1],))

        print("Training battery prediction model...")
        history = self.model.fit(
            X_train_scaled, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True)
            ]
        )

        # Evaluate
        test_loss, test_mae = self.model.evaluate(
            X_test_scaled, y_test, verbose=0)
        print("Test MAE: " + str(round(test_mae, 2)) + "%")

        # Save model and scaler
        self.model.save('battery_predictor.h5')
        joblib.dump(self.scaler, 'battery_scaler.pkl')
        print("Model saved as 'battery_predictor.h5'")
        print("Scaler saved as 'battery_scaler.pkl'")

        # Plot training history
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend()

        plt.tight_layout()
        plt.savefig('battery_training_history.png')
        plt.show()

        return test_mae

    def predict_battery(self, current_battery, current_speed, current_drain):
        """Predict next battery level"""
        if self.model is None:
            # Try to load existing model
            try:
                self.model = keras.models.load_model('battery_predictor.h5')
                self.scaler = joblib.load('battery_scaler.pkl')
                print("Loaded pre-trained model")
            except:
                print("No trained model found. Please train first.")
                return current_battery

        # Prepare input
        input_data = np.array(
            [[current_battery, current_speed, current_drain]])
        input_scaled = self.scaler.transform(input_data)

        # Predict
        prediction = self.model.predict(input_scaled, verbose=0)[0][0]
        return max(0, min(100, prediction))  # Clamp between 0-100


# Train the model with YOUR specific file path
if __name__ == "__main__":
    # YOUR SPECIFIC FILE PATH - Use raw string to handle backslashes
    csv_file_path = r'C:\Users\Pasindu Nirmal\OneDrive\Desktop\Heartfordshir Nebula\Second Year\First Semester\Behavioral Robotics\Assignments\Course_Work_02\CW_02 submit file\controllers\Battery_data_collect\battery_data.csv'

    print("Looking for data file: " + csv_file_path)

    # Check if file exists before training
    if not os.path.exists(csv_file_path):
        print("File not found! Please check:")
        print("  1. The file path is correct")
        print("  2. The file exists in that location")
        print("  3. You ran the data collection script first")

        # Show what files are in the directory
        directory = os.path.dirname(csv_file_path)
        if os.path.exists(directory):
            print("Files in directory " + directory + ":")
            for file in os.listdir(directory):
                print("   - " + file)
        else:
            print("Directory doesn't exist: " + directory)
    else:
        print("File found! Starting training...")
        predictor = BatteryPredictor()
        mae = predictor.train(csv_file_path)

        if mae is not None:
            # Test prediction
            test_battery = 45
            test_speed = 5.0
            test_drain = 1.2

            prediction = predictor.predict_battery(
                test_battery, test_speed, test_drain)
            print("Test Prediction: " + str(test_battery) +
                  "% -> " + str(round(prediction, 1)) + "%")

            print("Training completed successfully!")
            print("Model files created:")
            print("   - battery_predictor.h5")
            print("   - battery_scaler.pkl")
            print("   - battery_training_history.png")
