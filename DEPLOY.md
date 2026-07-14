# Deploying the Digital Twin (free)

The repo ships a `render.yaml` blueprint that stands up the whole app on
[Render](https://render.com)'s free tier — the FastAPI backend and the static
dashboard — in one step. No credit card required.

## What gets deployed

| Service | Type | What it does |
|---------|------|--------------|
| `turbojet-api` | Python web service | Trains the surrogate at build time, serves the API |
| `turbojet-dashboard` | Static site | The vanilla-JS dashboard, wired to the API automatically |

The dashboard's build step rewrites the backend URL into `js/api.js`, so the two
find each other with no manual editing.

## Steps (≈5 minutes)

1. **Push to GitHub** (already done if the repo is on GitHub):
   ```bash
   git push origin main
   ```

2. **Sign in to Render** — go to <https://render.com>, click *Get Started*, and
   sign in **with GitHub** (free, no card).

3. **Create the Blueprint** — in the Render dashboard:
   *New +* → *Blueprint* → pick this repository → Render reads `render.yaml` and
   shows the two services → *Apply*.

4. **Wait for the first build** — the backend installs deps and trains the model
   (~2–3 min); the dashboard builds in seconds. Watch the logs until both say
   *Live*.

5. **Open the dashboard** — click the `turbojet-dashboard` service; its URL is
   `https://turbojet-dashboard.onrender.com` (or similar). That link is public —
   share it with anyone.

## Good to know (free tier)

- **Cold starts.** The free backend **sleeps after ~15 min idle**. The first
  visitor after a nap waits ~30–60 s while it wakes and reloads the model, then
  it's fast. Fine for a demo; open the link a minute before showing it.
- **Redeploys.** Every `git push origin main` triggers an automatic rebuild.
- **No secrets.** Nothing here needs API keys or a database.

## Running locally instead

```bash
pip install -r requirements.txt
cd src/model && python train.py          # build the model
cd ../backend && uvicorn main:app --port 8000   # terminal 1
cd ../frontend && python -m http.server 8080    # terminal 2
# open http://localhost:8080/index.html
```
`js/api.js` auto-detects localhost, so the same code works locally and deployed.
