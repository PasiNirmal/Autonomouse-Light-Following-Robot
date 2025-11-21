# battery_controller_webots.py
import csv
import time
import numpy as np
import tensorflow as tf
import joblib
from controller import Robot, Camera

# --- Robot Setup ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Camera
camera = robot.getDevice("Astra rgb")
camera.enable(timestep)

# Parameters
MAX_SPEED = 6.4
CRUISING_SPEED = 6.0
SAFE_DISTANCE = 0.7

# Battery system
battery_capacity = 100
current_battery = 100
battery_drain_timer = time.time()
last_drain_rate = 1.0  # Default drain rate


# Load trained model
class BatteryManager:
    def __init__(self):
        self.model = tf.keras.models.load_model('battery_predictor.h5')
        self.scaler = joblib.load('battery_scaler.pkl')
        self.prediction_history = []

    def predict_future_battery(self, current_battery, current_speed, current_drain, steps_ahead=5):
        """Predict battery level several steps ahead"""
        future_battery = current_battery

        # Simulate multiple steps for better prediction
        for _ in range(steps_ahead):
            input_data = np.array([[future_battery, current_speed, current_drain]])
            input_scaled = self.scaler.transform(input_data)
            prediction = self.model.predict(input_scaled, verbose=0)[0][0]
            future_battery = max(0, min(100, prediction))

        return future_battery

    def calculate_speed_multiplier(self, current_battery, predicted_battery):
        """Calculate speed multiplier based on battery levels"""
        # Your plan: 40% to 25% range = 1.0 to 0.5 speed multiplier
        if current_battery > 40:
            return 1.0  # Full speed above 40%
        elif current_battery <= 25:
            return 0.5  # Minimum speed at/below 25%
        else:
            # Linear decrease from 40% to 25%
            # (current_battery - 25) / (40 - 25) gives 1.0 at 40, 0.0 at 25
            # We want 1.0 at 40, 0.5 at 25, so:
            ratio = (current_battery - 25) / (40 - 25)
            speed_multiplier = 0.5 + (0.5 * ratio)
            return max(0.5, min(1.0, speed_multiplier))


# Initialize battery manager
battery_manager = BatteryManager()


def obstacle_avoidance(ranges, speed_multiplier=1.0):
    """Navigate while avoiding obstacles with speed scaling"""
    if not ranges:
        return

    n = len(ranges)
    left_distance = min(ranges[0:n // 3]) if ranges[0:n // 3] else float('inf')
    front_distance = min(ranges[n // 3:2 * n // 3]) if ranges[n // 3:2 * n // 3] else float('inf')
    right_distance = min(ranges[2 * n // 3:]) if ranges[2 * n // 3:] else float('inf')

    base_speed = CRUISING_SPEED * speed_multiplier

    if front_distance < SAFE_DISTANCE:
        print("🚧 Obstacle detected - Turning...")
        if left_distance > right_distance:
            left_motor.setVelocity(-0.5 * base_speed)
            right_motor.setVelocity(0.5 * base_speed)
        else:
            left_motor.setVelocity(0.5 * base_speed)
            right_motor.setVelocity(-0.5 * base_speed)
    else:
        left_motor.setVelocity(base_speed)
        right_motor.setVelocity(base_speed)


# Main loop
print("🔋 Starting Smart Battery Management System...")

while robot.step(timestep) != -1:
    current_time = time.time()

    # === BATTERY MANAGEMENT ===
    # Calculate current speed for drain calculation
    current_speed = CRUISING_SPEED

    # Update battery every 2 seconds (slower drain in low battery)
    if current_time - battery_drain_timer > 2.0:
        # Adjust drain rate based on battery level (slower when battery low)
        if current_battery > 40:
            base_drain_rate = 1.0  # Normal drain
        elif current_battery > 25:
            # Gradually reduce drain rate from 40% to 25%
            ratio = (current_battery - 25) / (40 - 25)
            base_drain_rate = 0.5 + (0.5 * ratio)  # 1.0 at 40%, 0.5 at 25%
        else:
            base_drain_rate = 0.5  # Minimum drain rate

        current_battery = max(0, current_battery - base_drain_rate)
        last_drain_rate = base_drain_rate
        battery_drain_timer = current_time

        # Predict future battery
        predicted_battery = battery_manager.predict_future_battery(
            current_battery, current_speed, last_drain_rate
        )

        # Calculate speed multiplier based on your plan
        speed_multiplier = battery_manager.calculate_speed_multiplier(
            current_battery, predicted_battery
        )

        print(
            f"🔋 Battery: {current_battery:.1f}% | Predicted: {predicted_battery:.1f}% | Speed: {speed_multiplier:.1f}x | Drain: {last_drain_rate:.1f}%/2s")

    # === NAVIGATION ===
    ranges = robot.getDevice("Hokuyo URG-04LX-UG01").getRangeImage()

    # Get current speed multiplier (use latest calculated)
    speed_multiplier = battery_manager.calculate_speed_multiplier(current_battery, current_battery)

    # Navigate with speed scaling
    obstacle_avoidance(ranges, speed_multiplier)

    # Reset battery if empty (for continuous testing)
    if current_battery <= 0:
        current_battery = 100
        print("🔄 Battery reset to 100% for continuous testing")

    # Exit condition
    if robot.getTime() > 300:  # Run for 5 minutes
        print("⏰ Session complete!")
        break

print("🛑 Smart battery controller stopped.")