import os
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Input and output folders
input_folder = r'XXX'  # Folder containing k and v data files
output_folder = r'XXX'  # Path to save the result files
os.makedirs(output_folder, exist_ok=True)

# Definition of fuzzy variables and membership functions
density = ctrl.Antecedent(np.arange(0, 0.25, 0.001), 'density')
density['low'] = fuzz.gaussmf(density.universe, 0.0275, 0.02)
density['medium'] = fuzz.gaussmf(density.universe, 0.08125, 0.02)
density['high'] = fuzz.gaussmf(density.universe, 0.135, 0.02)

speed = ctrl.Antecedent(np.arange(0, 30.1, 0.1), 'speed')
speed['low'] = fuzz.gaussmf(speed.universe, 5, 3)
speed['medium'] = fuzz.gaussmf(speed.universe, 13, 3)
speed['high'] = fuzz.gaussmf(speed.universe, 22, 3)

congestion_prob = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'congestion_prob')
congestion_prob['low'] = fuzz.trimf(congestion_prob.universe, [0, 0, 0.6])
congestion_prob['medium'] = fuzz.trimf(congestion_prob.universe, [0.4, 0.6, 0.8])
congestion_prob['high'] = fuzz.trimf(congestion_prob.universe, [0.6, 0.85, 1.0])
congestion_prob['full'] = fuzz.trimf(congestion_prob.universe, [0.85, 1.0, 1.0])

# Define fuzzy inference rules
rules = [
    ctrl.Rule(density['high'] & speed['low'], congestion_prob['full']),
    ctrl.Rule(density['low'] & speed['low'], congestion_prob['medium']),
    ctrl.Rule(density['low'] & speed['medium'], congestion_prob['low']),
    ctrl.Rule(density['low'] & speed['high'], congestion_prob['low']),
    ctrl.Rule(density['medium'] & speed['low'], congestion_prob['high']),
    ctrl.Rule(density['medium'] & speed['medium'], congestion_prob['medium']),
    ctrl.Rule(density['medium'] & speed['high'], congestion_prob['low']),
    ctrl.Rule(density['high'] & speed['medium'], congestion_prob['high']),
    ctrl.Rule(density['high'] & speed['high'], congestion_prob['medium'])
]

# Create fuzzy control system and simulation engine
congestion_ctrl = ctrl.ControlSystem(rules)
congestion_simulation = ctrl.ControlSystemSimulation(congestion_ctrl)

# Extract output membership functions for classification
cp_uni = congestion_prob.universe
cp_low_mf = fuzz.trimf(cp_uni, [0, 0, 0.6])
cp_med_mf = fuzz.trimf(cp_uni, [0.4, 0.6, 0.8])
cp_high_mf = fuzz.trimf(cp_uni, [0.6, 0.85, 1.0])
cp_full_mf = fuzz.trimf(cp_uni, [0.85, 1.0, 1.0])
cp_labels = ['low', 'medium', 'high', 'full']
cp_mfs = [cp_low_mf, cp_med_mf, cp_high_mf, cp_full_mf]

# Batch processing for all scenarios
for i in range(1, 12):
    in_path = os.path.join(input_folder, f'Scenario_{i}.csv')
    df = pd.read_csv(in_path, encoding='gbk')

    cong_probs = []
    cong_labels = []
    for k, v in zip(df['K(t)'], df['xVelocity(t)']):
        congestion_simulation.input['density'] = k
        congestion_simulation.input['speed'] = v
        congestion_simulation.compute()
        p = congestion_simulation.output['congestion_prob']
        cong_probs.append(p)

        # Calculate membership degrees and select the label with the maximum degree
        degrees = [fuzz.interp_membership(cp_uni, mf, p) for mf in cp_mfs]
        cong_labels.append(cp_labels[int(np.argmax(degrees))])

    # Save predicted congestion probability and congestion level
    df['Congestion Probability P(A)'] = cong_probs
    df['Congestion Level'] = cong_labels

    out_name = f'Scenario_{i}_Processed.csv'
    df.to_csv(os.path.join(output_folder, out_name), index=False, encoding='utf-8-sig')
    print(f"Processed and saved: {out_name}")

print("Batch processing for all scenarios completed!")