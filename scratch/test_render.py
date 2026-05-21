import requests
import time

url = "https://ecabsd.onrender.com"

def test_prediction(pdb_id, chain_a, chain_b):
    print(f"\n--- Testing Predict for PDB: {pdb_id} ({chain_a} & {chain_b}) ---")
    predict_url = f"{url}/predict"
    data = {
        "pdb_id": pdb_id,
        "chain_a": chain_a,
        "chain_b": chain_b,
        "threshold": "auto",
        "mode": "threshold",
        "top_k_percent": 15.0
    }
    
    start = time.time()
    try:
        response = requests.post(predict_url, data=data, timeout=120)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            res_json = response.json()
            print("Prediction Status:", res_json.get("status"))
            print("GradCAM Allowed:", res_json.get("gradcam_allowed"))
            print("Total Residues:", res_json.get("total_residues"))
            print("Binding Ratio:", res_json.get("binding_ratio"))
            
            # Now try explanations if prediction succeeded
            test_explanation(pdb_id, chain_a, chain_b)
        else:
            print("Response:", response.text[:500])
    except Exception as e:
        print("Error during predict:", e)
    print(f"Time taken: {time.time() - start:.2f} seconds")

def test_explanation(pdb_id, chain_a, chain_b):
    print(f"--- Testing Explain/GradCAM for PDB: {pdb_id} ({chain_a} & {chain_b}) ---")
    explain_url = f"{url}/explain"
    data = {
        "pdb_id": pdb_id,
        "chain_a": chain_a,
        "chain_b": chain_b
    }
    start = time.time()
    try:
        response = requests.post(explain_url, data=data, timeout=120)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            res_json = response.json()
            print("Explain Status:", res_json.get("status"))
            print("GradCAM Available:", res_json.get("gradcam_available"))
            print("GradCAM Message:", res_json.get("gradcam_message"))
            print("Has GradCAM Image:", res_json.get("gradcam_image") is not None)
            print("Has Attention Image:", res_json.get("attention_image") is not None)
            print("Overlap Percentage:", res_json.get("overlap_percentage"))
        else:
            print("Response:", response.text[:500])
    except Exception as e:
        print("Error during explain:", e)
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    # Test 1DFJ E I
    test_prediction("1DFJ", "E", "I")
    
    # Wait a bit
    print("Waiting 10 seconds before next sample...")
    time.sleep(10)
    
    # Test 1BVN P T
    test_prediction("1BVN", "P", "T")
