# MAPSO Quick Start Guide

This guide will get you up and running with MAPSO in 5 minutes.

---

## 📋 Prerequisites

- Python 3.9 or higher
- pip package manager
- 4GB RAM minimum
- Windows, macOS, or Linux

---

## 🚀 Installation Steps

### Step 1: Set Up Python Environment

```bash
# Navigate to project directory
cd "C:\Users\Ujwal Mojidra\Desktop\AAI\project\Optimization Product Scheduling"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install MAPSO package
pip install -e .
```

This will install all required packages including:
- ortools (constraint programming)
- pandas, numpy (data manipulation)
- fastapi, streamlit (web frameworks)
- and more...

**Note**: Installation may take 5-10 minutes depending on your internet connection.

---

## ✅ Verify Installation

```bash
# Test imports
python -c "import mapso; print('MAPSO installed successfully!')"
python -c "from ortools.sat.python import cp_model; print('OR-Tools working!')"
```

If you see success messages, you're ready to go!

---

## 🎯 Run Your First Optimization

### Option 1: Run Demo Script (Recommended)

```bash
# Run the complete optimization demo
python scripts/run_optimization.py
```

This will:
1. Generate a synthetic dataset (20 SKUs, 5 machines, 50 jobs)
2. Create baseline schedules using 3 heuristics
3. Run CP-SAT optimization for 60 seconds
4. Display comparison results

**Expected runtime**: 1-2 minutes

**Expected output**:
```
========================================================================
MAPSO - Multi-Agent Production Scheduling Optimizer
========================================================================

[1/5] Generating synthetic dataset...
  Generated: 20 SKUs, 5 machines, 50 jobs

[2/5] Creating baseline schedules...
  fifo: 50 jobs scheduled
  edd: 50 jobs scheduled
  spt: 50 jobs scheduled

[3/5] Running CP-SAT optimization...
  CP-SAT: 50 jobs scheduled

[4/5] Calculating metrics...

[5/5] Results Comparison
========================================================================

schedule_name  n_scheduled_jobs  total_lateness_hours  n_late_jobs  ...
-------------  ----------------  --------------------  -----------  ...
fifo                         50                 245.3           23  ...
edd                          50                 187.6           18  ...
spt                          50                 312.8           28  ...
cpsat                        50                  98.4            9  ...

Improvements (CP-SAT vs EDD Baseline):
------------------------------------------------------------------------
  total_lateness_hours_improvement_pct: +47.6%
  n_late_jobs_improvement_pct: +50.0%
  ...
```

### Option 2: Python Script

Create a file `test_mapso.py`:

```python
from mapso.data.generator import DataGenerator
from mapso.optimization.layer1_cpsat import CPSATOptimizer
from datetime import datetime, timedelta

# Generate data
print("Generating dataset...")
generator = DataGenerator(random_seed=42)
dataset = generator.generate_complete_dataset(
    n_skus=10, n_machines=3, n_jobs=20
)

# Run optimization
print("Running optimization...")
optimizer = CPSATOptimizer(config={"timeout": 30})

schedule = optimizer.optimize(
    jobs=dataset["jobs"],
    machines=dataset["machines"],
    skus=dataset["skus"],
    changeover_matrix=dataset["changeover_matrix"],
    start_date=datetime.now(),
    end_date=datetime.now() + timedelta(days=7),
)

print(f"\n✓ Optimization complete!")
print(f"  Scheduled: {len(schedule.get_scheduled_jobs())} jobs")
print(f"  Lateness: {schedule.total_lateness:.1f} hours")
print(f"  On-time: {len(schedule.get_scheduled_jobs()) - schedule.n_late_jobs} jobs")
```

Run it:
```bash
python test_mapso.py
```

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'mapso'`

**Solution**:
```bash
# Make sure you're in the project directory
cd "C:\Users\Ujwal Mojidra\Desktop\AAI\project\Optimization Product Scheduling"

# Reinstall in editable mode
pip install -e .
```

### Issue: OR-Tools Installation Failed

**Problem**: Error installing ortools package

**Solution**:
```bash
# Try installing OR-Tools separately
pip install ortools==9.7.2996

# If still fails, try:
pip install --upgrade pip setuptools wheel
pip install ortools
```

### Issue: Import Errors

**Problem**: `ImportError: cannot import name 'X'`

**Solution**:
```bash
# Check Python version (must be 3.9+)
python --version

# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: Script Hangs

**Problem**: Optimization script runs indefinitely

**Solution**:
- Reduce problem size (fewer jobs/machines)
- Reduce timeout in config
- Check available RAM (close other programs)

---

## 📊 Understanding the Output

### Key Metrics Explained:

- **n_scheduled_jobs**: Number of jobs successfully scheduled
- **total_lateness_hours**: Sum of all late hours (0 = all on time)
- **n_late_jobs**: Count of jobs that missed due date
- **on_time_rate**: Percentage of jobs delivered on time (higher is better)
- **makespan_hours**: Total schedule duration
- **total_setup_time_hours**: Time spent on changeovers
- **avg_utilization**: Average machine utilization (0-1, higher is better)
- **total_cost**: Total production + setup + late penalty costs

### Good vs Bad Schedule:

**Good Schedule**:
- On-time rate > 80%
- Low lateness (< 100 hours for 50 jobs)
- High utilization (> 70%)
- Low setup time

**Bad Schedule**:
- On-time rate < 50%
- High lateness (> 200 hours)
- Low utilization (< 50%)
- Many late jobs

---

## 🎓 Next Steps

### 1. Experiment with Parameters

Modify `scripts/run_optimization.py`:

```python
# Change problem size
dataset = generator.generate_complete_dataset(
    n_skus=30,      # More product types
    n_machines=10,  # More machines
    n_jobs=100,     # More orders
)

# Change objective weights
objective_weights={
    "lateness": 0.6,    # Prioritize on-time delivery
    "setup_time": 0.1,
    "cost": 0.2,
    "energy": 0.1,
}

# Change timeout
optimizer = CPSATOptimizer(config={"timeout": 180})  # 3 minutes
```

### 2. Save and Load Datasets

```python
# Save generated dataset
generator.save_dataset(dataset, "data/synthetic/my_dataset")

# Load later
from mapso.data.loader import DataLoader
loader = DataLoader()
loaded_dataset = loader.load_dataset("data/synthetic/my_dataset")
```

### 3. Customize Agents

```python
from mapso.agents.orchestrator import OrchestratorAgent
from mapso.agents.capacity_machine import CapacityMachineAgent

# Create orchestrator
orchestrator = OrchestratorAgent()

# Register capacity agent
capacity_agent = CapacityMachineAgent()
orchestrator.register_agent(capacity_agent)

# Process schedule through agents
schedule = orchestrator.process(schedule)
```

### 4. Visualize Results

Coming soon: Streamlit dashboard with:
- Interactive Gantt charts
- KPI dashboards
- Agent insights
- Real-time optimization

---

## 📚 Learn More

- **README.md**: Full project documentation
- **docs/**: Technical documentation
- **scripts/**: More example scripts
- **tests/**: Test suite examples

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide first
2. Read error messages carefully
3. Check Python version: `python --version`
4. Check installed packages: `pip list | grep mapso`
5. Try reinstalling: `pip install -e . --force-reinstall`

---

## ✅ Success Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed successfully
- [ ] Demo script runs without errors
- [ ] See optimization results
- [ ] Understand key metrics

If all checked, you're ready to use MAPSO! 🎉

---

**Questions?** Check the main README.md or project documentation.
