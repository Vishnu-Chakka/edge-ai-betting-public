# 🚀 Deploy Backend & Configure Vercel - Quick Guide

Your Vercel frontend is live at:
**https://edge-ai-betting-public-jtusn6mgy-vishnus-projects-31db9273.vercel.app**

But it can't connect to the backend because the backend isn't deployed yet. Follow these steps:

---

## Step 1: Deploy Backend to Render (5 minutes)

### ⚡ RECOMMENDED: Manual Setup (Most Reliable)

1. **Go to Render**: https://dashboard.render.com/
2. **Sign up/Login** (it's free)
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**:
   - If this is your first time, authorize Render to access GitHub
   - Select your repository (edge-ai-betting)
5. **IMPORTANT - Configure the service EXACTLY like this**:
   - **Name**: `edge-ai-betting-backend`
   - **Region**: Oregon (US West)
   - **Branch**: `main`
   - **Root Directory**: `backend` ← **CRITICAL! Must be "backend"**
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: Select Python 3.11 or 3.10
6. **Add Environment Variables** (click "Advanced" → "Add Environment Variable"):

   Copy these EXACTLY:
   ```
   ANTHROPIC_API_KEY = your_anthropic_api_key_here

   ODDS_API_KEY = your_odds_api_key_here

   DATABASE_URL = sqlite+aiosqlite:///./edge_ai.db

   CORS_ORIGINS = https://edge-ai-betting-public-jtusn6mgy-vishnus-projects-31db9273.vercel.app,https://edge-ai-betting-public.vercel.app,http://localhost:3000

   ENVIRONMENT = production

   PYTHON_VERSION = 3.11
   ```

7. **Select Plan**: Free (starts automatically, spins down after 15 min of inactivity)
8. **Click "Create Web Service"**

⏱️ **Wait 2-3 minutes for deployment** - Watch the logs in the Render dashboard

9. **Copy your backend URL** when it says "Live" (will look like: `https://edge-ai-betting-backend.onrender.com`)

### ✅ Verify Backend is Working

Once deployed, test it:
```bash
curl https://your-backend-url.onrender.com/health
```
Should return: `{"status":"healthy"}`

---

### Option B: Using render.yaml (One-Click Deploy)

I've prepared a `render.yaml` file in the backend directory. You can use it for automatic deployment:

1. **Go to**: https://dashboard.render.com/
2. **Click "New +" → "Blueprint"**
3. **Connect your GitHub repository**
4. **Point to**: `backend/render.yaml`
5. **Add the secret environment variables** when prompted:
   - `ANTHROPIC_API_KEY`
   - `ODDS_API_KEY`
6. **Deploy**

---

## Step 2: Configure Vercel with Backend URL

Once your backend is deployed on Render, copy the URL (e.g., `https://edge-ai-betting-backend.onrender.com`) and run these commands:

```bash
# Add the backend URL to Vercel
vercel env add NEXT_PUBLIC_API_URL

# When prompted:
# - Enter the value: https://your-backend-url.onrender.com (paste your Render URL)
# - Select environments: Production, Preview, Development (select all)

# Also add as NEXT_PUBLIC_BACKEND_URL for compatibility
vercel env add NEXT_PUBLIC_BACKEND_URL

# When prompted:
# - Enter the value: https://your-backend-url.onrender.com (same URL)
# - Select environments: Production, Preview, Development (select all)
```

---

## Step 3: Redeploy Vercel

```bash
# Redeploy with the new environment variables
vercel --prod
```

---

## Step 4: Test Everything

1. **Test Backend Health**:
   ```bash
   curl https://your-backend-url.onrender.com/health
   ```
   Should return: `{"status":"healthy"}`

2. **Test Frontend**:
   - Visit: https://edge-ai-betting-public-jtusn6mgy-vishnus-projects-31db9273.vercel.app
   - Click "START CHATTING NOW"
   - Try sending a message
   - Should work exactly like localhost! ✅

---

## Quick Reference: Your API Keys

From your `backend/.env`:
- **ANTHROPIC_API_KEY**: (from your backend/.env file)
- **ODDS_API_KEY**: (from your backend/.env file)

⚠️ **Security Note**: These keys are in your `.env` file. Make sure `.env` is in `.gitignore` and never commit it!

---

## Troubleshooting

### ❌ "uvicorn command not found" error
**This means the Root Directory is wrong!**

✅ **FIX**: In Render dashboard:
1. Go to your service → Settings
2. Find "Root Directory"
3. **Change it to**: `backend`
4. Click "Save Changes"
5. Render will auto-redeploy with the correct path

### ❌ Frontend shows errors
- Check that environment variables are set in Vercel
- Verify the backend URL is correct (no trailing slash)
- Make sure you redeployed after adding env vars

### ❌ Backend deployment fails
- Check Render logs for errors
- Verify all environment variables are set correctly
- Make sure Root Directory is set to `backend`
- Ensure Python version is 3.10 or 3.11

### ❌ CORS errors
- Verify `CORS_ORIGINS` includes your Vercel URL
- Must include both the deployment URL and production URL
- No trailing slashes in URLs

---

## Alternative: Use Railway Instead of Render

If you prefer Railway:

1. Go to: https://railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Set Root Directory: `backend`
5. Add same environment variables as above
6. Deploy

Railway URL will be like: `https://edge-ai-betting-backend.up.railway.app`

---

That's it! Your Vercel site will match localhost perfectly! 🎉
