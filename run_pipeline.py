import os
import shutil
import subprocess
import argparse
import itertools

# --- CONFIGURATION ---
BASE_DIR = "/mnt/e/AI/khanh"
WAV2UNIT_SCRIPT = os.path.join(BASE_DIR, "src/Wav2Unit/infer.py")
UNIT2UNIT_SCRIPT = os.path.join(BASE_DIR, "src/Unit2Unit/infer_v4_dur.py")
UNIT2WAV_SCRIPT = os.path.join(BASE_DIR, "src/Unit2Wav/infer.py")

# Intermediate and Final paths
WAV2UNIT_SOURCE_INPUT = os.path.join(BASE_DIR, "final/wav2unit/source/input/input.wav")
WAV2UNIT_SOURCE_OUTPUT = os.path.join(BASE_DIR, "final/wav2unit/source/predicted_unit.txt")
UNIT2WAV_TARGET_OUTPUT = os.path.join(BASE_DIR, "final/unit2wav/target/predicted_wav/result_vn.wav")

def deduplicate_units(unit_str):
    """'15 15 15 20' -> '15 20'"""
    units = unit_str.strip().split()
    if not units: return ""
    dedup_units = [k for k, g in itertools.groupby(units)]
    return " ".join(dedup_units)

def run_step(description, command):
    print(f"\n{'='*20} {description} {'='*20}")
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Unified Speech-to-Speech Translation Pipeline")
    parser.add_argument("--input", required=True, help="Path to input English audio file (.wav)")
    parser.add_argument("--output", default="output_vi.wav", help="Path to save the final Vietnamese audio file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    # Step 0: Ensure directories exist and copy input
    os.makedirs(os.path.dirname(WAV2UNIT_SOURCE_INPUT), exist_ok=True)
    shutil.copy(args.input, WAV2UNIT_SOURCE_INPUT)

    # Step 1: Wav -> Raw Units (English)
    run_step("Wav2Unit (English)", ["python", WAV2UNIT_SCRIPT, "--lang", "source"])

    # Step 2: Deduplicate English Units
    print("\n>>> Deduplicating English Units...")
    with open(WAV2UNIT_SOURCE_OUTPUT, 'r') as f:
        raw_units_en = f.read().strip()
    dedup_units_en = deduplicate_units(raw_units_en)
    
    # Overwrite the predicted_unit.txt with the dedup version for Unit2Unit
    # Note: infer_v4_dur.py reads from stdin if --input is not provided, 
    # but we'll use the --input argument for clarity.
    
    # Step 3: Translate English (Dedup) -> Vietnamese (Expanded)
    run_step("Unit2Unit (EN -> VI)", ["python", UNIT2UNIT_SCRIPT, "--input", dedup_units_en])

    # Step 4: Vietnamese (Expanded) -> Wav
    run_step("Unit2Wav (Vietnamese)", ["python", UNIT2WAV_SCRIPT, "--lang", "target"])

    # Step 5: Final Cleanup and Move
    if os.path.exists(UNIT2WAV_TARGET_OUTPUT):
        shutil.copy(UNIT2WAV_TARGET_OUTPUT, args.output)
        print(f"\n{'='*60}")
        print(f"SUCCESS! Final Vietnamese audio saved to: {args.output}")
        print(f"{'='*60}")
    else:
        print("\nERROR: Final audio file was not generated.")

if __name__ == "__main__":
    main()
