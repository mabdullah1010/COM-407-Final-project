import random

import libpyAI as ai

BACK_THRESHOLD = 100
MAX_SPEED = 3
SHOT_ALERT_THRESHOLD = 60
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sklearn.preprocessing._data as _skdata
StandardScaler = _skdata.StandardScaler

torch.serialization.add_safe_globals([StandardScaler])

class Model(nn.Module):
    def __init__(self, in_features, h1=256, h2=128, h3 = 96, h4=64,out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)


        self.out = nn.Linear(h4, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        x = self.out(x)
        return x

checkpoint = torch.load("shooter.pth", map_location="cpu", weights_only=False)


scaler = checkpoint["scaler"]
in_features = scaler.mean_.shape[0]

model = Model(in_features=in_features)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model + scaler loaded successfully! in_features =", in_features)

def predict_action(raw_features):
    x = np.array(raw_features).reshape(1, -1)

    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).numpy()[0]

    preds = (probs >= 0.5).astype(int)
    return probs, preds

def norm_distance(x):
    return x / 1500


def norm_speed(x):
    return x / 20


def norm_angle(x):
    return x / 360


def norm_angle_diff(x):
    return x / 180


def norm_shot_alert(x):
    if x is None:
        return 0
    if x <= 80:
        return 1
    if x >= 200:
        return 0
    return (200 - x) / 120


def angle_diff(ang1, ang2):
    return ((ang1 - ang2 + 180) % 360) - 180


def AI_loop():
    agent_heading = int(ai.selfHeadingDeg())
    agent_tracking = int(ai.selfTrackingDeg())
    agent_speed = int(ai.selfSpeed())
    agent_aim = int(ai.aimdir(0))
    shot_alert = int(ai.shotAlert(0))

    ai.setTurnSpeedDeg(20)

    # Release keys
    ai.thrust(0)
    ai.turnLeft(0)
    ai.turnRight(0)

    action_thrust = 0
    action_left = 0
    action_right = 0
    action_shoot = 0

    feelers = []
    for i in range(0, 360, 1):
        feelers.append(ai.wallFeeler(1500, agent_heading + i))

    # Determined feelers
    front_dist = feelers[0]
    back_dist = feelers[180]
    back_left_dist = feelers[175]
    back_right_dist = feelers[185]

    closest_dist = min(feelers)
    furthest_dist = max(feelers)

    closest_angle = (agent_heading + feelers.index(closest_dist)) % 360
    furthest_angle = (agent_heading + feelers.index(furthest_dist)) % 360

    heading_tracking_diff = angle_diff(agent_tracking, agent_heading)
    closest_diff = angle_diff(closest_angle, agent_heading)
    furthest_diff = angle_diff(furthest_angle, agent_heading)
    aim_diff = angle_diff(agent_aim, agent_heading)

    goingBackwards = False
    if abs(heading_tracking_diff) > 80:
        goingBackwards = True

    backdanger = (back_dist < BACK_THRESHOLD or
                  back_left_dist < BACK_THRESHOLD or
                  back_right_dist < BACK_THRESHOLD) and goingBackwards

    example_input = [
            norm_angle(agent_heading),
            norm_angle(agent_tracking),
            norm_speed(agent_speed),
            norm_angle(agent_aim),
            norm_shot_alert(shot_alert),
            norm_distance(front_dist),
            norm_distance(back_dist),
            norm_distance(back_left_dist),
            norm_distance(back_right_dist),
            norm_distance(closest_dist),
            norm_distance(furthest_dist),
            norm_angle(closest_angle),
            norm_angle(furthest_angle),
            norm_angle_diff(heading_tracking_diff),
            norm_angle_diff(closest_diff),
            norm_angle_diff(furthest_diff),
            norm_angle_diff(aim_diff)]

    probs, preds = predict_action(example_input)
    print(preds)

    if ai.selfSpeed() ==0:
        if random.randint(1,10)==4:
            preds[0] = 1

    if action_shoot == preds[3]:
        ai.fireShot()
    ai.turnLeft(preds[1])
    ai.turnRight(preds[2])
    ai.thrust(preds[0])


ai.start(AI_loop, ["-name", "trained_shooter", "-join", "localhost"])
