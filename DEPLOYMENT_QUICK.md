# Quick Deployment Solution

## The Problem
Your Vercel frontend is trying to call `/api` endpoints, but Vercel only hosts frontend code. There's no backend server to handle API requests.

## The Solution (5 minutes)

### Option 1: Deploy Backend to Render (Recommended)
1. **Go to Render**: https://render.com
2. **Create Account** (free)
3. **New Web Service** → Connect GitHub
4. **Select Repository**: `tigee1311/edge-ai-betting`
5. **Configure**:
   - Name: `edge-ai-betting-backend`
   - Root Directory: `backend/`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
6. **Environment Variables**:
   - `ANTHROPIC_API_KEY=your_anthropic_api_key_here`
   - `ODDS_API_KEY=your_odds_api_key_here`
7. **Deploy** (takes 2-3 minutes)

### Option 2: Use Railway (Alternative)
1. **Go to Railway**: https://railway.app
2. **Connect GitHub**
3. **Select Repository**: `tigee1311/edge-ai-betting`
4. **Set Root Directory**: `backend/`
5. **Add Environment Variables**:
   - `ANTHROPIC_API_KEY=your_anthropic_api_key_here`
   - `ODDS_API_KEY=your_odds_api_key_here`
6. **Deploy**

### Configure Vercel
1. **Go to Vercel Dashboard**
2. **Your Project** → Settings → Environment Variables
3. **Add Variable**:
   - Name: `NEXT_PUBLIC_BACKEND_URL`
   - Value: `https://your-backend-url.com` (from Render/Railway)
4. **Redeploy**

## Test
- Backend: `https://your-backend.onrender.com/health`
- Frontend: Your Vercel URL
- Chat should work!

## Cost
- Render: Free tier (with limitations) or $7/month
- Railway: Free tier available or $5/month
- Vercel: Free for frontend

This will fix the 404 error completely!