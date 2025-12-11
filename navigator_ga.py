
import json
import time
import random
import argparse
import os
import threading
import math

import libpyAI as ai

GENE_SIZE = 5
CHROMOSOME_SIZE = 11
DEFAULT_POPULATION_SIZE = 128
MUTATION_RATE = 1/100.0
DEFAULT_GENERATION_INTERVAL = 10 * 60  # 10 minutes
POPULATION_FILENAME = "navigator_pop.txt"
ARCHIVE_FILENAME = "navigator_fitness_results.txt"
UPDATES_FILENAME = "navigator_fitness_updates.jsonl"
EPS = 1e-9
RUNS_PER_CHROM = 2
ELITE_COUNT = 2

def generate_gene(size=GENE_SIZE):
    return [random.randint(0,1) for _ in range(size)]

def generate_chromosome(gsize=GENE_SIZE, csize=CHROMOSOME_SIZE):
    return [generate_gene(gsize) for _ in range(csize)]

def generate_population(size=DEFAULT_POPULATION_SIZE):
    return [generate_chromosome() for _ in range(size)]

def save_population(pop, fname=POPULATION_FILENAME):
    with open(fname, "w") as f:
        json.dump(pop, f)

def load_population(fname=POPULATION_FILENAME):
    with open(fname, "r") as f:
        return json.load(f)

def binary_list_to_int(lst):
    if not lst:
        return 0
    return int(''.join(map(str, lst)), 2)

def chromo_decoder(chromosome):
    front_dist_th = binary_list_to_int(chromosome[0]) * 20
    max_speed_th = binary_list_to_int(chromosome[1])
    back_th = binary_list_to_int(chromosome[2]) * 5
    head_track_diff_th = binary_list_to_int(chromosome[3]) * 4
    furthest_diff_th1 = binary_list_to_int(chromosome[4])
    shot_alert_th = binary_list_to_int(chromosome[5]) * 3
    enemy_dist_th = binary_list_to_int(chromosome[6]) * 4
    closest_dist_th1 = binary_list_to_int(chromosome[7]) * 3
    closest_dist_th2 = binary_list_to_int(chromosome[8]) * 4
    furthest_diff_th2 = binary_list_to_int(chromosome[9])
    aim_diff_th = binary_list_to_int(chromosome[10])
    return [
        front_dist_th, max_speed_th, back_th, head_track_diff_th, furthest_diff_th1,
        shot_alert_th, enemy_dist_th, closest_dist_th1, closest_dist_th2, furthest_diff_th2, aim_diff_th
    ]

def load_archive(fname):
    archive = {}
    if not os.path.exists(fname):
        return archive
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            decoded = obj.get("decoded_params")
            if decoded is None:
                continue
            key = tuple(decoded)
            avg = float(obj.get("avg_fitness", 0.0))
            runs = int(obj.get("runs", 0))
            archive[key] = (avg, runs)
    return archive

def write_archive(fname, archive):
    with open(fname, "w") as f:
        for key, (avg, runs) in archive.items():
            obj = {"decoded_params": list(key), "avg_fitness": float(avg), "runs": int(runs)}
            f.write(json.dumps(obj) + "\n")

def load_updates(fname):
    if not os.path.exists(fname):
        return []
    updates = []
    with open(fname, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            updates.append(json.loads(s))
    os.remove(fname)
    return updates

def merge_updates_into_archive(archive, updates):
    for u in updates:
        decoded = u.get("decoded_params")
        if decoded is None:
            continue
        key = tuple(decoded)
        avg_u = float(u.get("avg_fitness", 0.0))
        runs_u = int(u.get("runs", 0))
        if key in archive:
            avg_a, runs_a = archive[key]
            combined_runs = runs_a + runs_u
            if combined_runs > 0:
                combined_avg = (avg_a * runs_a + avg_u * runs_u) / combined_runs
            else:
                combined_avg = 0.0
            archive[key] = (combined_avg, combined_runs)
        else:
            archive[key] = (avg_u, runs_u)
    return archive

def compute_archive_mean(archive):
    vals = [v for (v, r) in archive.values() if r > 0]
    return (sum(vals) / len(vals)) if vals else 0.0

def init_fitness_list_from_archive(population, archive, mean_avg):
    baseline = mean_avg if mean_avg > 0 else 1.0
    fitness_list = []
    for chrom in population:
        key = tuple(chromo_decoder(chrom))
        if key in archive:
            avg, runs = archive[key]
            fitness = float(avg)
        else:
            fitness = float(baseline)
        fitness_list.append(fitness + EPS)
    return fitness_list

def roulette_select_index(fitnesses):
    total = sum(fitnesses)
    if total <= 0.0:
        return random.randrange(len(fitnesses))
    pick = random.random() * total
    cum = 0.0
    for i, f in enumerate(fitnesses):
        cum += f
        if pick <= cum:
            return i
    return len(fitnesses) - 1

def gene_wise_intragene_uniform_crossover(parent_a, parent_b):
    child = []
    for ga, gb in zip(parent_a, parent_b):
        new_gene = [ga[i] if random.random() < 0.5 else gb[i] for i in range(len(ga))]
        child.append(new_gene)
    return child

def mutate_chromosome(chromosome, mutation_rate=MUTATION_RATE):
    for gi in range(len(chromosome)):
        gene = chromosome[gi]
        for bi in range(len(gene)):
            if random.random() < mutation_rate:
                gene[bi] = 1 - gene[bi]
    return chromosome

def make_next_generation(population, fitness_list, pop_size):
    next_pop = []
    sorted_idx = sorted(range(len(population)), key=lambda i: fitness_list[i], reverse=True)
    elites = [population[i] for i in sorted_idx[:ELITE_COUNT]]
    next_pop.extend(elites)

    while len(next_pop) < pop_size:
        ia = roulette_select_index(fitness_list)
        ib = roulette_select_index(fitness_list)
        parent_a = population[ia]
        parent_b = population[ib]
        child = gene_wise_intragene_uniform_crossover(parent_a, parent_b)
        child = mutate_chromosome(child)
        next_pop.append(child)

    while len(next_pop) < pop_size:
        next_pop.append(generate_chromosome())

    return next_pop[:pop_size]

def select_chromosome_by_roulette_for_assignment(fitness_list_local, eval_counts_local, runs_needed):
    eligible = [i for i in range(len(fitness_list_local)) if eval_counts_local[i] < runs_needed]
    if not eligible:
        return None
    elig_fitnesses = [fitness_list_local[i] for i in eligible]
    total = sum(elig_fitnesses)
    if total <= 0.0:
        return random.choice(eligible)
    pick = random.random() * total
    cum = 0.0
    for idx, f in zip(eligible, elig_fitnesses):
        cum += f
        if pick <= cum:
            return idx
    return eligible[-1]

n_pop = load_population(POPULATION_FILENAME)

archive = load_archive(ARCHIVE_FILENAME)
mean_avg = compute_archive_mean(archive)

fitness_list = init_fitness_list_from_archive(n_pop, archive, mean_avg)

eval_counts = [0] * len(n_pop)
sum_fitness = [0.0] * len(n_pop)

def manager_thread_fn(interval, pop_size):
    global n_pop, fitness_list, archive, mean_avg, eval_counts, sum_fitness
    gen = 0
    print(f"[MANAGER] starting: interval={interval}s pop_size={pop_size} archive_entries={len(archive)}")
    while True:
        print(f"[MANAGER] sleeping {interval}s until generation {gen}")
        time.sleep(interval)

        updates = load_updates(UPDATES_FILENAME)
        if updates:
            print(f"[MANAGER] merging {len(updates)} updates")
            archive = merge_updates_into_archive(archive, updates)
            write_archive(ARCHIVE_FILENAME, archive)
            mean_avg = compute_archive_mean(archive)
        else:
            print("[MANAGER] no updates to merge")

        if len(fitness_list) != len(n_pop):
            fitness_list = init_fitness_list_from_archive(n_pop, archive, mean_avg)

        best_fit = max(fitness_list) if fitness_list else 0.0
        avg_fit = (sum(fitness_list)/len(fitness_list)) if fitness_list else 0.0
        print(f"[MANAGER] BEFORE gen {gen}: best={best_fit:.3f} avg={avg_fit:.3f} archive_entries={len(archive)}")

        new_pop = make_next_generation(n_pop, fitness_list, pop_size)

        save_population(new_pop, POPULATION_FILENAME)
        print(f"[MANAGER] wrote new population for generation {gen} -> {POPULATION_FILENAME}")

        fitness_list = init_fitness_list_from_archive(new_pop, archive, mean_avg)

        eval_counts = [0] * len(new_pop)
        sum_fitness = [0.0] * len(new_pop)

        n_pop = new_pop
        gen += 1

current_life_start_time = None
current_chromosome = None
current_chromo_idx = None
previous_score = 0
shots_fired = 0

_previous_x = None
_previous_y = None
_total_distance = 0.0

def write_update_for_chromosome(chromo_idx):
    decoded = chromo_decoder(n_pop[chromo_idx])
    total = sum_fitness[chromo_idx]
    runs = eval_counts[chromo_idx]
    avg = float(total / runs) if runs > 0 else 0.0
    obj = {"index": chromo_idx, "avg_fitness": avg, "runs": runs, "decoded_params": decoded}
    with open(UPDATES_FILENAME, "a") as f:
        f.write(json.dumps(obj) + "\n")
    print(f"[AI] appended update for chromosome {chromo_idx}: avg={avg:.4f} runs={runs}")

def compute_distance_fitness(total_distance):
    return float(total_distance)

def AI_loop():
    global current_life_start_time, current_chromosome, current_chromo_idx
    global previous_score, shots_fired, fitness_list, eval_counts, sum_fitness
    global _previous_x, _previous_y, _total_distance

    score = ai.selfScore()
    alive = (ai.selfAlive() == 1)
    current_time = time.time()

    if alive and current_life_start_time is None:
        current_life_start_time = current_time
        shots_fired = 0
        previous_score = score

        try:
            _previous_x = ai.selfX()
            _previous_y = ai.selfY()
        except Exception:
            _previous_x = None
            _previous_y = None
        _total_distance = 0.0

        idx = select_chromosome_by_roulette_for_assignment(fitness_list, eval_counts, RUNS_PER_CHROM)
        if idx is None:
            current_chromosome = None
            current_chromo_idx = None
            print("[AI] No eligible chromosome available for assignment (all done). Agent idle.")
        else:
            current_chromo_idx = idx
            current_chromosome = n_pop[current_chromo_idx]
            print("=== NEW LIFE START ===")
            print(f"[AI] start_time: {time.ctime(current_time)}  chromosome_index: {current_chromo_idx}")
            print(f"[AI] seed fitness: {fitness_list[current_chromo_idx]:.4f}")

    if current_chromosome is not None and alive:
        chromo_params = chromo_decoder(current_chromosome)

        front_dist_th = chromo_params[0]
        max_speed_th = chromo_params[1]
        back_th = chromo_params[2]
        head_track_diff_th = chromo_params[3]
        furthest_diff_th1 = chromo_params[4]
        shot_alert_th = chromo_params[5]
        enemy_dist_th = chromo_params[6]
        closest_dist_th1 = chromo_params[7]
        closest_dist_th2 = chromo_params[8]
        furthest_diff_th2 = chromo_params[9]
        aim_diff_th = chromo_params[10]

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

        heading_tracking_diff = ((agent_tracking - agent_heading + 180) % 360) - 180
        closest_diff = ((closest_angle - agent_heading + 180) % 360) - 180
        furthest_diff = ((furthest_angle - agent_heading + 180) % 360) - 180
        aim_diff = ((agent_aim - agent_heading + 180) % 360) - 180

        goingBackwards = False
        if abs(heading_tracking_diff) > 80:
            goingBackwards = True

        backdanger = (back_dist < back_th or
                      back_left_dist < back_th or
                      back_right_dist < back_th) and goingBackwards

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
        elif abs(furthest_diff) > furthest_diff_th1 and furthest_diff > 0:
            action_left = 1
        elif abs(furthest_diff) > furthest_diff_th2 and furthest_diff <= 0:
            action_right = 1

        # Firing Logic
        if abs(aim_diff) <= 5 and not agent_aim == -1 and ai.wallFeeler(1500, agent_aim) > ai.enemyDistance(0):
            action_shoot = 1

        if action_shoot == 1:
            ai.fireShot()
            shots_fired += 1

        if agent_speed == 0:
            if random.randint(0,50) == 4:
                action_thrust = 1

        ai.turnLeft(action_left)
        ai.turnRight(action_right)
        ai.thrust(action_thrust)

        try:
            x = ai.selfX()
            y = ai.selfY()
            if _previous_x is None or _previous_y is None:
                _previous_x = x
                _previous_y = y
            else:
                dx = x - _previous_x
                dy = y - _previous_y
                step = math.hypot(dx, dy)
                _total_distance += step
                _previous_x = x
                _previous_y = y
        except Exception:
            pass

    if (not alive) and (current_life_start_time is not None) and (current_chromosome is not None):
        life_duration = current_time - current_life_start_time
        score_gain = score - previous_score

        fitness = compute_distance_fitness(_total_distance)

        sum_fitness[current_chromo_idx] += fitness
        eval_counts[current_chromo_idx] += 1

        avg_so_far = sum_fitness[current_chromo_idx] / eval_counts[current_chromo_idx]
        fitness_list[current_chromo_idx] = avg_so_far + EPS

        print("=== LIFE ENDED ===")
        print(f"[AI] chromosome {current_chromo_idx} life_duration={life_duration:.2f}s distance={_total_distance:.2f} fitness+={fitness:.2f}")
        print(f"[AI] chromosome {current_chromo_idx} accumulated fitness={sum_fitness[current_chromo_idx]:.2f} runs={eval_counts[current_chromo_idx]}/{RUNS_PER_CHROM}")

        if eval_counts[current_chromo_idx] >= RUNS_PER_CHROM:
            write_update_for_chromosome(current_chromo_idx)

        current_life_start_time = None
        current_chromosome = None
        current_chromo_idx = None
        previous_score = 0
        shots_fired = 0
        _previous_x = None
        _previous_y = None
        _total_distance = 0.0

def main():
    global n_pop, fitness_list, archive, mean_avg, eval_counts, sum_fitness

    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=DEFAULT_GENERATION_INTERVAL, help='seconds between generations')
    parser.add_argument('--pop', type=int, default=len(n_pop), help='population size (regenerates if different)')
    args = parser.parse_args()


    if args.pop != len(n_pop):
        n_pop = generate_population(args.pop)
        save_population(n_pop, POPULATION_FILENAME)
        archive = load_archive(ARCHIVE_FILENAME)
        mean_avg = compute_archive_mean(archive)
        fitness_list = init_fitness_list_from_archive(n_pop, archive, mean_avg)
        eval_counts = [0] * len(n_pop)
        sum_fitness = [0.0] * len(n_pop)
        print(f"[MAIN] regenerated population with size {args.pop}")

    mgr = threading.Thread(target=manager_thread_fn, args=(args.interval, args.pop), daemon=True)
    mgr.start()

    print("[MAIN] starting AI loop (blocks). Watch the console logs for GA activity.")
    ai.start(AI_loop, ["-name", "main", "-join", "localhost"])

if __name__ == "__main__":
    main()
