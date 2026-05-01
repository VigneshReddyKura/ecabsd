import os
import json

def cleanup_results(results_dir='results/', ratio_threshold=0.40):
    """
    Scans the results directory and removes JSON files that exceed the binding ratio threshold.
    """
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' not found.")
        return

    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not files:
        print(f"No JSON files found in '{results_dir}'.")
        return

    deleted_count = 0
    for file_name in files:
        path = os.path.join(results_dir, file_name)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            count = data.get('binding_residues_count', 0)
            total = data.get('total_residues', 1)
            ratio = count / total
            
            if ratio > ratio_threshold:
                print(f"Deleting {file_name} (Ratio: {ratio:.2%}) - Overprediction detected.")
                os.remove(path)
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print(f"\nCleanup complete. Deleted {deleted_count} file(s).")

if __name__ == "__main__":
    cleanup_results()
