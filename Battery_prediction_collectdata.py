# data_collection_webots.py
import csv
import time
import os
from controller import Robot

# Initialize robot
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Motors
left_motor = robot.getDevice("wheel_left_joint")
right_motor = robot.getDevice("wheel_right_joint")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# Battery simulation parameters
battery_capacity = 100
current_battery = 100
battery_drain_timer = time.time()
last_speed = 0

# Create CSV file
csv_file = 'battery_data.csv'
file_exists = os.path.isfile(csv_file)

with open(csv_file, 'a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['battery_level', 'motor_speed', 'battery_drain_rate', 'speed_multiplier'])

print("🔋 Starting battery data collection...")
print("📊 Data will be saved to battery_data.csv")

while robot.step(timestep) != -1:
    current_time = time.time()

    # Simulate different speed patterns
    simulation_time = robot.getTime()

    # Vary speed in patterns (0.1 to 1.0 of MAX_SPEED)
    if simulation_time < 30:
        speed_multiplier = 1.0  # Full speed
    elif simulation_time < 60:
        speed_multiplier = 0.7  # Medium speed
    elif simulation_time < 90:
        speed_multiplier = 0.3  # Slow speed
    else:
        speed_multiplier = 0.8  # Mixed speed

    MAX_SPEED = 6.4
    current_speed = MAX_SPEED * speed_multiplier

    left_motor.setVelocity(current_speed)
    right_motor.setVelocity(current_speed)

    # Battery drain based on speed (faster speed = faster drain)
    if current_time - battery_drain_timer > 1.0:  # Update every second
        # Drain rate: 0.5% at low speed, up to 2% at high speed
        drain_rate = 0.5 + (1.5 * speed_multiplier)
        current_battery = max(0, current_battery - drain_rate)
        battery_drain_timer = current_time

        # Calculate actual drain rate for this period
        actual_drain_rate = drain_rate

        # Save data to CSV
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                current_battery,
                current_speed,
                actual_drain_rate,
                speed_multiplier
            ])

        print(f"Battery: {current_battery:.1f}% | Speed: {speed_multiplier:.1f}x | Drain: {actual_drain_rate:.1f}%/s")

        # Reset when battery reaches 0
        if current_battery <= 0:
            current_battery = 100
            print("🔄 Resetting battery to 100%")

    # Stop after collecting sufficient data
    if simulation_time > 120:  # 2 minutes of data collection
        print("✅ Data collection complete!")
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        break

print(f"📁 Data saved to {csv_file}")