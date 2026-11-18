# PathSnap — two-laptop deployment (Windows)
- **Laptop 1 = backend** (abhin's laptop): FastAPI + model + HMM + evaluation data (incl. ground truth, never sent out).
- **Laptop 2 = frontend** (paya5am's laptop): Streamlit UI only. Talks to Laptop 1 over the network.
Both laptops on the SAME network. Campus Wi-Fi often blocks laptop-to-laptop traffic ("client isolation"):
if Laptop 2 can't reach Laptop 1, put both on one phone hotspot.

## Laptop 1 — backend (Anaconda Prompt, in the repo folder)
```bat
conda create -n pathsnap python=3.11 -y
conda activate pathsnap
pip install -r requirements_api.txt
```
1. **Evaluation trips.** Until demo day use the student-test folder; on demo day point to the evaluator folder
   (the folder that directly contains 00000015, 00000017, ...):
   ```bat
   set PATHSNAP_EVAL_DIR=C:\Users\abhin\Desktop\PES\sem_5\ML\JackfruitHackathon\Student folder\Student folder\student_test
   ```
   (PowerShell: `$env:PATHSNAP_EVAL_DIR="C:\...\student_test"`)
2. **Model (M8).** Without artifacts the API runs the PLACEHOLDER (fine for UI development). To install the real model,
   run the export here (~2 min) — see artifacts/README.md:
   ```bat
   python scripts\export_model.py --data-root "C:\Users\abhin\Desktop\PES\sem_5\ML\JackfruitHackathon\Student folder\Student folder" --params scripts\inputs\best_params.json --hmm-config scripts\inputs\hmm_config.json --out artifacts
   ```
3. **Start the API** (keep this window visible: the evaluator must see these logs):
   ```bat
   python -m api.main
   ```
   Expect: `model: hist_gb_tuned_ll (trained)` and `evaluation folder: ... | 10 trips`.
4. **Firewall** (once, Administrator PowerShell):
   ```powershell
   New-NetFirewallRule -DisplayName "PathSnap API 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
   ```
   Or click "Allow access" on the Windows prompt when the API first starts (tick Private AND Public networks).
5. **Find the IP** to give to Laptop 2: `ipconfig` → "Wireless LAN adapter Wi-Fi" → IPv4 Address (e.g. 192.168.1.23).
6. **Test locally:** in a second window: `python scripts\smoke_test_api.py http://localhost:8000` → `ALL CHECKS PASSED`.

## Laptop 2 — frontend (paya5am)
```bat
conda create -n pathsnap-ui python=3.11 -y
conda activate pathsnap-ui
pip install -r requirements_ui.txt
set PATHSNAP_API_URL=http://192.168.1.23:8000
streamlit run ui\app.py
```
The browser opens the UI. Sidebar → "Check connection" → must say "Connected · 10 evaluation trips". A red
"PLACEHOLDER" banner means Laptop 1 has no real model yet. Test the network path first with
`python scripts\smoke_test_api.py http://192.168.1.23:8000` from Laptop 2.

## Demo-day checklist
- [ ] Laptop 1: evaluator trip folder copied; `PATHSNAP_EVAL_DIR` set; `set PATHSNAP_ALLOW_PLACEHOLDER=0`; API started; logs visible
- [ ] /health shows `model_status: trained`, decoder `tuned`, no version warnings
- [ ] Laptop 2: UI connected, lists exactly the 10 evaluator trips; run 2–3 trips before the evaluator arrives
- [ ] During the demo: evaluator picks a trip → UI → API (show the RESULT line in the Laptop 1 console) → map + metrics
- [ ] Screenshots for the report §7: UI with a result, Laptop 1 console with the request logs, logs/requests.csv
- [ ] Nothing is cached or hard-coded: every click recomputes candidates → classifier → HMM on Laptop 1
