"""
Advanced Concurrent SALBP Optimization Script for NVIDIA cuOpt
Features: True MILP with integer variables, stable multi-process thread handling,
robust status parsing, stateful checkpointing, real-time logging, and expenditure tracking.
"""

import os
import sys
import math
import time
import psutil
import pandas as pd
import threading
import multiprocessing
import concurrent.futures

HAS_NVML = False
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    try:
        from nvidia_ml_py import pynvml
        HAS_NVML = True
    except ImportError:
        HAS_NVML = False

from cuopt.linear_programming.problem import Problem, MINIMIZE, INTEGER
from cuopt.linear_programming.solver_settings import SolverSettings

# Global execution settings
time_limit = 900  # seconds
CONFIG = {
    "time_limit": time_limit,
    "mip_relative_gap": 0.01,
    "mip_scaling": 2,
    "num_cpu_threads": 7,   # 7 dedicated threads per instance
    "mip_probing": True,    # Enabled for deep tree pruning
    "max_workers": 2        # 2 workers * 7 threads = 14 cores (zero oversubscription)
}


class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Using 'a' (append) so checkpoint restarts don't overwrite previous logs
        self.log_file = open(filepath, "a", buffering=1, encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()  # FORCES immediate writing to disk, bypassing OS buffers

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.flush()
        self.log_file.close()


def get_resource_usage():
    proc = psutil.Process(os.getpid())
    ram_gb = proc.memory_info().rss / (1024 ** 3)
    cpu_pct = proc.cpu_percent(interval=0.05)

    vram_gb = 0.0
    if HAS_NVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_gb = gpu_info.used / (1024 ** 3)
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return ram_gb, vram_gb, cpu_pct


def parse_in2_file(file_path):
    with open(file_path, "r", encoding="latin-1") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    num_tasks = int(raw_lines[0])
    task_times = {}
    precedences = []
    idx, task_id = 1, 1

    while idx < len(raw_lines) and task_id <= num_tasks:
        val = raw_lines[idx]
        if val.startswith("-1"): break
        parts = val.replace(",", " ").split()
        if len(parts) == 1:
            task_times[task_id] = int(parts[0])
            task_id += 1
        elif len(parts) >= 2:
            task_times[int(parts[0])] = int(parts[1])
            task_id += 1
        idx += 1

    while idx < len(raw_lines) and (raw_lines[idx] == "-1" or raw_lines[idx] == "-1, -1"):
        idx += 1

    while idx < len(raw_lines):
        line = raw_lines[idx].replace(",", " ")
        parts = line.split()
        if len(parts) >= 2:
            u, v = int(parts[0]), int(parts[1])
            if u == -1 or v == -1: break
            precedences.append((u, v))
        idx += 1

    return num_tasks, task_times, precedences


def load_all_instances(excel_path, base_dataset_dir):
    print("Mapping .IN2 precedence graphs across directory tree...")
    
    # Map Excel names to actual .IN2 filenames
    alias_map = {
        'arcus1': 'arc83',
        'arcus2': 'arc111',
        'bowman': 'bowman8',
        'heskiaoff': 'heskia',
        'kilbridge': 'kilbrid',
        'sawyer': 'sawyer30',
        'tonge': 'tonge70'
    }
    
    in2_map = {}
    for root, dirs, files in os.walk(base_dataset_dir):
        for file in files:
            if file.lower().endswith('.in2'):
                g_name = os.path.splitext(file)[0].strip().lower()
                in2_map[g_name] = os.path.join(root, file)
                
    print(f"Discovered {len(in2_map)} unique precedence graphs.")
    
    xls = pd.ExcelFile(excel_path)
    sheets = [s for s in xls.sheet_names if "SALBP" in s.upper()]
    all_instances = []
    
    for sheet in sheets:
        try:
            df = pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception as e:
            print(f"Warning: Could not read sheet {sheet}: {e}")
            continue
        
        sheet_upper = sheet.upper()
        if "SALBP-1" in sheet_upper:
            data = df[[0, 1, 2]].dropna()
            for _, row in data.iterrows():
                graph_name = str(row[0]).strip().lower()
                search_name = alias_map.get(graph_name, graph_name)
                
                if search_name not in in2_map: continue
                try:
                    c = int(float(row[1]))
                    m_star = int(float(row[2]))
                    all_instances.append(("SALBP-1", in2_map[search_name], graph_name.upper(), c, m_star))
                except (ValueError, TypeError): continue
                
        elif "SALBP-2" in sheet_upper:
            data = df[[0, 1, 2]].dropna()
            for _, row in data.iterrows():
                graph_name = str(row[0]).strip().lower()
                search_name = alias_map.get(graph_name, graph_name)
                
                if search_name not in in2_map: continue
                try:
                    m = int(float(row[1]))
                    c_star = int(float(row[2]))
                    all_instances.append(("SALBP-2", in2_map[search_name], graph_name.upper(), m, c_star))
                except (ValueError, TypeError): continue
                
        elif "SALBP-E" in sheet_upper:
            data = df[[0, 4, 5]].dropna()
            for _, row in data.iterrows():
                graph_name = str(row[0]).strip().lower()
                search_name = alias_map.get(graph_name, graph_name)
                
                if search_name not in in2_map: continue
                try:
                    m_star = int(float(row[4]))
                    c_star = int(float(row[5]))
                    all_instances.append(("SALBP-E", in2_map[search_name], graph_name.upper(), c_star, m_star))
                except (ValueError, TypeError): continue
            
    print(f"Successfully staged {len(all_instances)} valid benchmark instances.")
    return all_instances


def solve_instance(args):
    prob_type, in2_path, graph_name, input_val, lit_target = args
    num_tasks, task_times, precedences = parse_in2_file(in2_path)
    
    problem = Problem(f"{prob_type}_{graph_name}")
    task_ids = list(task_times.keys())
    
    # ---------------------------------------------------------
    # Formulation 1: SALBP-1 & SALBP-E (Given C, Minimize m)
    # ---------------------------------------------------------
    if prob_type in ["SALBP-1", "SALBP-E"]:
        cycle_time = input_val
        max_stations = len(task_times)
        
        x, y = {}, {}
        for i in task_ids:
            for k in range(1, max_stations + 1):
                x[(i, k)] = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name=f"x_{i}_{k}")
        for k in range(1, max_stations + 1):
            y[k] = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name=f"y_{k}")
            
        problem.setObjective(sum(y[k] for k in range(1, max_stations + 1)), sense=MINIMIZE)
        
        for i in task_ids:
            problem.addConstraint(sum(x[(i, k)] for k in range(1, max_stations + 1)) == 1, name=f"assign_{i}")
        for k in range(1, max_stations + 1):
            station_load = sum(task_times[i] * x[(i, k)] for i in task_ids)
            problem.addConstraint(station_load <= cycle_time * y[k], name=f"cycle_limit_{k}")
            
        for pred, succ in precedences:
            if pred in task_times and succ in task_times:
                pred_pos = sum(k * x[(pred, k)] for k in range(1, max_stations + 1))
                succ_pos = sum(k * x[(succ, k)] for k in range(1, max_stations + 1))
                problem.addConstraint(pred_pos <= succ_pos, name=f"prec_{pred}_{succ}")

    # ---------------------------------------------------------
    # Formulation 2: SALBP-2 (Given m, Minimize C)
    # ---------------------------------------------------------
    elif prob_type == "SALBP-2":
        m = input_val
        sum_t = sum(task_times.values())
        max_t = max(task_times.values())
        
        C = problem.addVariable(lb=max_t, ub=sum_t, vtype=INTEGER, name="C")
        x = {}
        for i in task_ids:
            for k in range(1, m + 1):
                x[(i, k)] = problem.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name=f"x_{i}_{k}")
                
        problem.setObjective(C, sense=MINIMIZE)
        
        for i in task_ids:
            problem.addConstraint(sum(x[(i, k)] for k in range(1, m + 1)) == 1, name=f"assign_{i}")
        for k in range(1, m + 1):
            station_load = sum(task_times[i] * x[(i, k)] for i in task_ids)
            problem.addConstraint(station_load - C <= 0, name=f"cycle_limit_{k}")
            
        for pred, succ in precedences:
            if pred in task_times and succ in task_times:
                pred_pos = sum(k * x[(pred, k)] for k in range(1, m + 1))
                succ_pos = sum(k * x[(succ, k)] for k in range(1, m + 1))
                problem.addConstraint(pred_pos - succ_pos <= 0, name=f"prec_{pred}_{succ}")

    settings = SolverSettings()
    settings.set_parameter("time_limit", CONFIG["time_limit"])
    settings.set_parameter("mip_relative_gap", CONFIG["mip_relative_gap"])
    settings.set_parameter("mip_scaling", CONFIG["mip_scaling"])
    settings.set_parameter("num_cpu_threads", CONFIG["num_cpu_threads"])
    settings.set_parameter("mip_probing", CONFIG["mip_probing"])

    start_t = time.time()
    solve_result = problem.solve(settings)
    solve_dur = time.time() - start_t

    # Robust Status Resolution
    status = "Unknown"
    if hasattr(problem, "Status") and problem.Status is not None:
        status = getattr(problem.Status, "name", str(problem.Status))
    elif solve_result is not None:
        status = str(solve_result)

    # Robust Objective Extraction with NaN & None safety check
    val_cuopt = None
    if hasattr(problem, "ObjValue") and problem.ObjValue is not None:
        try:
            val_f = float(problem.ObjValue)
            if not math.isnan(val_f):
                val_cuopt = int(round(val_f))
        except (ValueError, TypeError):
            pass
    
    ram, vram, cpu_pct = get_resource_usage()

    return prob_type, graph_name, input_val, lit_target, val_cuopt, status, solve_dur, ram, vram, cpu_pct


peak_ram_gb = 0.0
peak_vram_gb = 0.0
avg_cpu_util = 0.0
avg_gpu_util = 0.0
total_cpu_core_seconds = 0.0
total_gpu_active_seconds = 0.0
stop_monitor = False

def resource_monitor_daemon():
    global peak_ram_gb, peak_vram_gb, avg_cpu_util, avg_gpu_util
    global total_cpu_core_seconds, total_gpu_active_seconds, stop_monitor
    
    cpu_hist = []
    gpu_hist = []
    
    if HAS_NVML:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
    psutil.cpu_percent(interval=None)
    total_cores = psutil.cpu_count(logical=True)
    poll_interval = 0.5
        
    while not stop_monitor:
        total_ram = sum(p.memory_info().rss for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()) / (1024**3)
        if total_ram > peak_ram_gb: 
            peak_ram_gb = total_ram
        
        cpu_pct = psutil.cpu_percent(interval=None)
        cpu_hist.append(cpu_pct)
        total_cpu_core_seconds += (cpu_pct / 100.0) * total_cores * poll_interval
        
        if HAS_NVML:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                vram = mem_info.used / (1024 ** 3)
                if vram > peak_vram_gb: 
                    peak_vram_gb = vram
                
                gpu_hist.append(util.gpu)
                total_gpu_active_seconds += (util.gpu / 100.0) * poll_interval
            except Exception:
                pass
                
        time.sleep(poll_interval)
        
    if HAS_NVML:
        pynvml.nvmlShutdown()
        
    avg_cpu_util = sum(cpu_hist) / len(cpu_hist) if cpu_hist else 0.0
    avg_gpu_util = sum(gpu_hist) / len(gpu_hist) if gpu_hist else 0.0


def main():
    multiprocessing.set_start_method('spawn', force=True)

    out_log_path = "results/salbp_full_dataset_benchmark.txt"
    
    # --- CHECKPOINT RECOVERY ---
    completed_keys = set()
    if os.path.exists(out_log_path):
        with open(out_log_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[0].startswith("SALBP"):
                    # parts[0]=Type, parts[1]=Graph, parts[2]=Input (C=... or m=...)
                    completed_keys.add((parts[0], parts[1], parts[2]))

    tee = TeeLogger(out_log_path)
    sys.stdout = tee

    base_dataset_dir = "Scholl_dataset"
    excel_path = os.path.join(base_dataset_dir, "SALBP data sets.xlsx")
    
    raw_tasks = load_all_instances(excel_path, base_dataset_dir)
    tasks_args = []
    
    for t in raw_tasks:
        prob_type = t[0]
        g_name = t[2].upper()
        input_str = f"C={t[3]}" if prob_type in ["SALBP-1", "SALBP-E"] else f"m={t[3]}"
        
        # Only queue instance if it hasn't been completed in a previous run
        if (prob_type, g_name[:12], input_str) not in completed_keys:
            tasks_args.append(t)

    print(f"\n[CHECKPOINT] Discovered {len(completed_keys)} completed instances.")
    print(f"[CHECKPOINT] {len(tasks_args)} instances remaining in queue.")
    
    if not tasks_args:
        print("All instances completed! Exiting.")
        return

    global stop_monitor
    monitor_thread = threading.Thread(target=resource_monitor_daemon)
    monitor_thread.start()

    print("\n" + "=" * 120)
    print(f"Executing Batch with Time Limit: {CONFIG['time_limit']} seconds")
    print(f"{'Type':<10}{'Graph':<14}{'Input':<10}{'Lit Opt':<10}{'cuOpt':<10}{'Diff':<8}{'Status':<14}{'Time(s)':<10}{'CPU(%)':<10}{'RAM(GB)':<10}{'VRAM(GB)':<10}")
    print("=" * 120)

    total_start_time = time.time()
    workers_to_use = min(CONFIG["max_workers"], len(tasks_args))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_to_use) as executor:
        # Submit all tasks asynchronously
        future_to_task = {executor.submit(solve_instance, arg): arg for arg in tasks_args}
        
        # Yield and print results instantly as they complete, ignoring queue order
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                prob_type, g_name, input_val, lit_opt, cuopt_val, status, dur, ram, vram, cpu_pct = result
                
                input_str = f"C={input_val}" if prob_type in ["SALBP-1", "SALBP-E"] else f"m={input_val}"
                diff_str = str(cuopt_val - lit_opt) if cuopt_val is not None else "N/A"
                val_str = str(cuopt_val) if cuopt_val is not None else "N/A"
                
                print(f"{prob_type:<10}{g_name[:12]:<14}{input_str:<10}{lit_opt:<10}{val_str:<10}{diff_str:<8}{status:<14}{dur:<10.2f}{cpu_pct:<10.1f}{ram:<10.2f}{vram:<10.2f}")
            except Exception as exc:
                print(f"An instance generated an exception: {exc}")

    total_time = time.time() - total_start_time
    
    stop_monitor = True
    monitor_thread.join()

    print("=" * 120)
    print("EXECUTION & RESOURCE SUMMARY (CURRENT RUN):")
    print(f"Total Wall-Clock Time:    {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Peak System RAM Usage:    {peak_ram_gb:.2f} GB (Aggregated across active processes)")
    print(f"Peak GPU VRAM Usage:      {peak_vram_gb:.2f} GB")
    print(f"Average CPU Util:         {avg_cpu_util:.1f}%")
    print(f"Average GPU Util:         {avg_gpu_util:.1f}%")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("COMPUTATION EXPENDITURE (CURRENT RUN):")
    print(f"Cumulative CPU Work:      {total_cpu_core_seconds:.2f} Core-Seconds ({total_cpu_core_seconds/3600:.4f} Core-Hours)")
    print(f"Cumulative GPU Work:      {total_gpu_active_seconds:.2f} GPU-Seconds ({total_gpu_active_seconds/3600:.4f} GPU-Hours)")
    print("- - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    print("SOLVER & BATCHING CONFIGURATION:")
    print("Formulation Bounding:     Unconstrained (max_stations = N)")
    print(f"Parallel CPU Probing:     Enabled (mip_probing = {CONFIG['mip_probing']}, num_cpu_threads = {CONFIG['num_cpu_threads']})")
    print(f"GPU Concurrent Batching:  Enabled (ProcessPoolExecutor max_workers = {workers_to_use})")
    print(f"Instance Time Limit:      {CONFIG['time_limit']} seconds")
    print(f"MIP Relative Gap Target:  {CONFIG['mip_relative_gap']}")
    print("=" * 120)

    tee.close()
    sys.stdout = tee.terminal
    print(f"\nExecution results successfully saved to {out_log_path}")


if __name__ == "__main__":
    main()