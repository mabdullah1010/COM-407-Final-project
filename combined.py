import random
import libpyAI as ai

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sklearn.preprocessing._data as _skdata
StandardScaler = _skdata.StandardScaler
torch.serialization.add_safe_globals([StandardScaler])


class Model(nn.Module):
    def __init__(self, in_features, h1=196, h2=128, h3=96, h4=64, out_features=4):
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

class GatingNetwork(nn.Module):
    def __init__(self, in_features, hidden=128, n_experts=3, temp=1.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.logits = nn.Linear(hidden, n_experts)
        self.temp = temp
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.logits(x) / max(self.temp, 1e-6)
        return F.softmax(logits, dim=-1)

def norm_distance(x): return x / 1500
def norm_speed(x): return x / 20
def norm_angle(x): return x / 360
def norm_angle_diff(x): return x / 180
def norm_shot_alert(x):
    if x is None: return 0
    if x <= 80: return 1
    if x >= 200: return 0
    return (200 - x) / 120
def angle_diff(ang1, ang2):
    return ((ang1 - ang2 + 180) % 360) - 180

def load_expert(checkpoint_path, h1_override=None):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    scaler = ckpt["scaler"]
    in_features = scaler.mean_.shape[0]
    h1 = h1_override if h1_override is not None else 196
    model = Model(in_features=in_features, h1=h1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, scaler, in_features

nav_model, nav_scaler, in_features = load_expert("navigator.pth", h1_override=196)
shoot_model, shoot_scaler, _ = load_expert("shooter.pth", h1_override=256)
thrust_model, thrust_scaler, _ = load_expert("thruster.pth", h1_override=256)

for nm, sc in [("nav", nav_scaler), ("shoot", shoot_scaler), ("thrust", thrust_scaler)]:
    if sc.mean_.shape[0] != in_features:
        raise RuntimeError(f"{nm} scaler in_features {sc.mean_.shape[0]} != expected {in_features}")

gating = GatingNetwork(in_features=in_features, hidden=128, n_experts=3, temp=1.0)
gating.eval()

def expert_probs_from_model(model, scaler, raw_features):
    x = np.array(raw_features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.sigmoid(logits).numpy()[0]
    return probs

def mixed_action_probs(raw_features):
    p_nav = expert_probs_from_model(nav_model, nav_scaler, raw_features)
    p_shoot = expert_probs_from_model(shoot_model, shoot_scaler, raw_features)
    p_thrust = expert_probs_from_model(thrust_model, thrust_scaler, raw_features)

    x = np.array(raw_features).reshape(1, -1)
    x_scaled_for_gating = nav_scaler.transform(x)
    x_tensor = torch.tensor(x_scaled_for_gating, dtype=torch.float32)
    with torch.no_grad():
        weights = gating(x_tensor).numpy()[0]

    final_probs = weights[0]*p_nav + weights[1]*p_shoot + weights[2]*p_thrust
    return final_probs, (p_nav, p_shoot, p_thrust), weights

def AI_loop():
    agent_heading = int(ai.selfHeadingDeg())
    agent_tracking = int(ai.selfTrackingDeg())
    agent_speed = int(ai.selfSpeed())
    agent_aim = int(ai.aimdir(0))
    shot_alert = int(ai.shotAlert(0))
    ai.setTurnSpeedDeg(20)
    ai.thrust(0); ai.turnLeft(0); ai.turnRight(0)

    feelers = [ai.wallFeeler(1500, agent_heading + i) for i in range(0, 360, 1)]
    front_dist = feelers[0]; back_dist = feelers[180]
    back_left_dist = feelers[175]; back_right_dist = feelers[185]
    closest_dist = min(feelers); furthest_dist = max(feelers)
    closest_angle = (agent_heading + feelers.index(closest_dist)) % 360
    furthest_angle = (agent_heading + feelers.index(furthest_dist)) % 360

    heading_tracking_diff = angle_diff(agent_tracking, agent_heading)
    closest_diff = angle_diff(closest_angle, agent_heading)
    furthest_diff = angle_diff(furthest_angle, agent_heading)
    aim_diff = angle_diff(agent_aim, agent_heading)

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
        norm_angle_diff(aim_diff)
    ]

    final_probs, per_expert_probs, weights = mixed_action_probs(example_input)
    final_preds = (final_probs >= 0.5).astype(int)
    if ai.selfSpeed() == 0 and random.randint(1, 10) == 4:
        final_preds[0] = 1

    if final_preds[3] == 1: ai.fireShot()
    ai.turnLeft(int(final_preds[1])); ai.turnRight(int(final_preds[2])); ai.thrust(int(final_preds[0]))

ai.start(AI_loop, ["-name", "mixture_bot", "-join", "localhost"])
