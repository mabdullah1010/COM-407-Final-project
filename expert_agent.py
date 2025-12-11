
import libpyAI as ai

BACK_THRESHOLD = 100
MAX_SPEED = 3
SHOT_ALERT_THRESHOLD = 60

HEADER = [
    "thrust",
    "turnLeft",
    "turnRight",
    "shoot",
    "agent_heading",
    "agent_tracking",
    "agent_speed",
    "agent_aim",
    "shot_alert",
    "front_dist",
    "back_dist",
    "back_left_dist",
    "back_right_dist",
    "closest_dist",
    "furthest_dist",
    "closest_angle",
    "furthest_angle",
    "heading_tracking_diff",
    "closest_diff",
    "furthest_diff",
    "aim_diff"
]

def norm_distance(x):
    return x/1500

def norm_speed(x):
    return x/20

def norm_angle(x):
    return x/360

def norm_angle_diff(x):
    return x/180

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

    ai.headlessMode()


    agent_heading = int(ai.selfHeadingDeg())
    agent_tracking = int(ai.selfTrackingDeg())
    agent_speed = int(ai.selfSpeed())
    agent_aim = int(ai.aimdir(0))
    shot_alert = int(ai.shotAlert(0))

    # Standardize turn speed
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

    # Identify the closest and furthest distances in the set of feelers
    closest_dist = min(feelers)
    furthest_dist = max(feelers)

    # Identify the closest and furthest angle values which are also indexes
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

    # Thrust Logic

    if front_dist > 400 and agent_speed <= MAX_SPEED and abs(furthest_diff) < 15: # Case: 0
        action_thrust = 1

    elif backdanger:  # Case: 1
        action_thrust = 1

    elif 0 < shot_alert < SHOT_ALERT_THRESHOLD: # Case: 2
        action_thrust = 1

    # Turn Logic

    if ai.enemyDistance(0) < 80 and aim_diff > 0 and agent_speed < 2: # Case: 0
        action_left = 1
    
    elif ai.enemyDistance(0) < 80 and aim_diff <= 0 and agent_speed < 2: # Case: 1
        action_right = 1 

    elif closest_dist < 70 and closest_diff > 0 and agent_speed != 0: # Case: 2
        action_right = 1 

    elif closest_dist < 70 and closest_diff <= 0 and agent_speed != 0: # Case: 3
        action_left = 1 

    elif closest_dist < 100 and furthest_diff > 0: # Case: 4
        action_left = 1 

    elif closest_dist < 100 and furthest_diff <= 0: # Case: 5
        action_right = 1 

    elif 0 < shot_alert < SHOT_ALERT_THRESHOLD and furthest_diff > 0: # Case: 6
        action_left = 1

    elif 0 < shot_alert < SHOT_ALERT_THRESHOLD and furthest_diff <= 0: # Case: 7
        action_right = 1

    elif agent_aim != -1 and abs(aim_diff) > 1 and aim_diff > 0: # Case: 8
        action_left = 1

    elif agent_aim != -1 and abs(aim_diff) > 1 and aim_diff <= 0: # Case: 9
        action_right = 1

    elif abs(furthest_diff) > 10 and furthest_diff > 0: # Case: 10
        action_left = 1

    elif abs(furthest_diff) > 10 and furthest_diff <= 0: # Case: 11
        action_right = 1
    
    # Firing Logic
    if abs(aim_diff) <= 5 and not agent_aim == -1 and ai.wallFeeler(1500, agent_aim) > ai.enemyDistance(0): # Case: 0, fire iff closest enemy without 5 degrees of aim and not behind a wall
            action_shoot = 1

    if action_shoot == 1:
        ai.fireShot()
    ai.turnLeft(action_left)
    ai.turnRight(action_right)
    ai.thrust(action_thrust)

    # if ai.selfAlive() == 1:
    #     write_data(
    #         LOG_FILE,
    #         [
    #             action_thrust,
    #             action_left,
    #             action_right,
    #             action_shoot,
    #             norm_angle(agent_heading),
    #             norm_angle(agent_tracking),
    #             norm_speed(agent_speed),
    #             norm_angle(agent_aim),
    #             norm_shot_alert(shot_alert),
    #             norm_distance(front_dist),
    #             norm_distance(back_dist),
    #             norm_distance(back_left_dist),
    #             norm_distance(back_right_dist),
    #             norm_distance(closest_dist),
    #             norm_distance(furthest_dist),
    #             norm_angle(closest_angle),
    #             norm_angle(furthest_angle),
    #             norm_angle_diff(heading_tracking_diff),
    #             norm_angle_diff(closest_diff),
    #             norm_angle_diff(furthest_diff),
    #             norm_angle_diff(aim_diff),
    #         ],
    #     )


ai.start(AI_loop,["-name","ExpertAgent","-join","localhost"])
