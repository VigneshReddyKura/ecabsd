import asyncio
import httpx
import os
import sys

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from web.app import app

async def run_local_test(pdb_id, chain_a, chain_b):
    print(f"\n==========================================")
    print(f"Testing {pdb_id} ({chain_a} & {chain_b}) locally")
    print(f"==========================================")
    
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=120.0) as client:
        # 1. Test Predict
        data = {
            "pdb_id": pdb_id,
            "chain_a": chain_a,
            "chain_b": chain_b,
            "threshold": "auto",
            "mode": "threshold",
            "top_k_percent": 15.0
        }
        print("Sending predict request...")
        res = await client.post("/predict", data=data)
        print("Predict response status:", res.status_code)
        
        if res.status_code == 200:
            json_data = res.json()
            print("Predict status:", json_data.get("status"))
            print("GradCAM allowed:", json_data.get("gradcam_allowed"))
            print("Total residues:", json_data.get("total_residues"))
            print("Binding ratio:", json_data.get("binding_ratio"))
            print("Number of predicted residues:", len(json_data.get("residues", [])))
            
            # 2. Test Explain
            print("\nSending explain request...")
            explain_data = {
                "pdb_id": pdb_id,
                "chain_a": chain_a,
                "chain_b": chain_b
            }
            res_explain = await client.post("/explain", data=explain_data)
            print("Explain response status:", res_explain.status_code)
            if res_explain.status_code == 200:
                explain_json = res_explain.json()
                print("Explain status:", explain_json.get("status"))
                print("GradCAM available:", explain_json.get("gradcam_available"))
                print("GradCAM message:", explain_json.get("gradcam_message"))
                print("Overlap percentage:", explain_json.get("overlap_percentage"))
            else:
                print("Explain response text:", res_explain.text[:1000])
        else:
            print("Predict response text:", res.text[:1000])

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Test 1DFJ E I
        loop.run_until_complete(run_local_test("1DFJ", "E", "I"))
    except Exception as e:
        print("Exception for 1DFJ E I:", e)
        
    try:
        # Test 1BVN P T
        loop.run_until_complete(run_local_test("1BVN", "P", "T"))
    except Exception as e:
        print("Exception for 1BVN P T:", e)
