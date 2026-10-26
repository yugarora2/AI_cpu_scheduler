"""Your original interactive menu, now using the headless schedulers (prints Gantt chart + table)."""
from schedulers import fcfs, sjf, srtf, priority_np, priority_p, round_robin, mlfq


def get_processes():
    n = int(input("Enter number of processes: "))
    if n <= 0:
        raise ValueError("Number of processes must be positive.")
    processes = []
    for i in range(n):
        print(f"\nProcess {i + 1}")
        pid = input("PID: ").strip()
        at = int(input("Arrival Time: "))
        bt = int(input("Burst Time: "))
        priority = int(input("Priority (lower number = higher priority): "))
        if not pid or at < 0 or bt <= 0:
            raise ValueError("PID must not be empty, AT >= 0, BT > 0.")
        processes.append({"pid": pid, "at": at, "bt": bt, "priority": priority})
    if len({p["pid"] for p in processes}) != len(processes):
        raise ValueError("Process IDs must be unique.")
    return processes


def main():
    print("CPU Scheduling Algorithms")
    print("1. FCFS\n2. SJF (Non-preemptive)\n3. SRTF (Preemptive SJF)")
    print("4. Priority (Non-preemptive)\n5. Priority (Preemptive)\n6. Round Robin\n7. MLFQ")
    choice = int(input("\nChoose algorithm: "))
    processes = get_processes()
    if choice == 1:
        fcfs(processes, verbose=True)
    elif choice == 2:
        sjf(processes, verbose=True)
    elif choice == 3:
        srtf(processes, verbose=True)
    elif choice == 4:
        priority_np(processes, verbose=True)
    elif choice == 5:
        priority_p(processes, verbose=True)
    elif choice == 6:
        round_robin(processes, int(input("Enter time quantum: ")), verbose=True)
    elif choice == 7:
        mlfq(processes, verbose=True)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
