
import libpyAI as ai
#from logger import init_logger, write_data

BACK_THRESHOLD = 100
MAX_SPEED = 3
SHOT_ALERT_THRESHOLD = 60


gold = [280, 6, 145, 28, 10, 81, 116, 93, 120, 13, 5]

HEADER = [
    "thrust", "turnLeft", "turnRight", "shoot",
    "agent_heading", "agent_tracking", "agent_speed", "agent_aim",
    "shot_alert", "front_dist", "back_dist", "back_left_dist", "back_right_dist",
    "closest_dist", "furthest_dist", "closest_angle", "furthest_angle",
    "heading_tracking_diff", "closest_diff", "furthest_diff", "aim_diff"
]

#LOG_FILE = init_logger("ExpertAgent", HEADER)



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
    global gold
    #ai.headlessMode()


    agent_heading = int(ai.selfHeadingDeg())
    agent_tracking = int(ai.selfTrackingDeg())
    agent_speed = int(ai.selfSpeed())
    agent_aim = int(ai.aimdir(0))
    shot_alert = int(ai.shotAlert(0))

    ai.setTurnSpeedDeg(20)

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

    (
        front_dist_th, max_speed_th, back_th, head_track_diff_th,
        furthest_diff_th1, shot_alert_th, enemy_dist_th,
        closest_dist_th1, closest_dist_th2,
        furthest_diff_th2, aim_diff_th
    ) = gold

    goingBackwards = abs(heading_tracking_diff) > 80

    backdanger = (
        (back_dist < back_th or back_left_dist < back_th or back_right_dist < back_th)
        and goingBackwards
    )


    if front_dist > front_dist_th and agent_speed <= max_speed_th and abs(furthest_diff) < furthest_diff_th1:
        action_thrust = 1

    elif backdanger:
        action_thrust = 1

    elif 0 < shot_alert < shot_alert_th:
        action_thrust = 1


    if ai.enemyDistance(0) < enemy_dist_th and aim_diff > 0 and agent_speed < 2:
        action_left = 1

    elif ai.enemyDistance(0) < enemy_dist_th and aim_diff <= 0 and agent_speed < 2:
        action_right = 1

    elif closest_dist < closest_dist_th1 and closest_diff > 0 and agent_speed != 0:
        action_right = 1

    elif closest_dist < closest_dist_th1 and closest_diff <= 0 and agent_speed != 0:
        action_left = 1

    elif closest_dist < closest_dist_th2 and furthest_diff > 0:
        action_left = 1

    elif closest_dist < closest_dist_th2 and furthest_diff <= 0:
        action_right = 1

    elif 0 < shot_alert < shot_alert_th and furthest_diff > 0:
        action_left = 1

    elif 0 < shot_alert < shot_alert_th and furthest_diff <= 0:
        action_right = 1

    elif agent_aim != -1 and abs(aim_diff) > aim_diff_th and aim_diff > 0:
        action_left = 1

    elif agent_aim != -1 and abs(aim_diff) > aim_diff_th and aim_diff <= 0:
        action_right = 1

    elif abs(furthest_diff) > furthest_diff_th2 and furthest_diff > 0:
        action_left = 1

    elif abs(furthest_diff) > furthest_diff_th2 and furthest_diff <= 0:
        action_right = 1

    # ------------------------ Shooting Logic ------------------------

    if (
        abs(aim_diff) <= aim_diff_th
        and agent_aim != -1
        and ai.wallFeeler(1500, agent_aim) > ai.enemyDistance(0)
    ):
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
    #

ai.start(AI_loop, ["-name", "Thruster", "-join", "localhost"])
