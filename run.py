import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_step(name, module_name):
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {name}")
    print(f"COMMAND: {sys.executable} -m {module_name}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"\nFAILED at step: {name}")
        sys.exit(result.returncode)

def main():
    run_step("Get Data", "data.get_data") 
    run_step("Data Pipeline", "src.data_pipeline")
    run_step("Training", "src.train")
    run_step("Evaluation", "src.eval")
    run_step("NLP Experiment", "src.nlp_classifier")
    run_step("RL Simulation", "src.rl_agent")
    print("\nAll pipeline steps completed successfully.")

if __name__ == "__main__":
    main()