# Edge AI Betting - Deployment Guide

## Overview

This application consists of two parts:
1. **Frontend** (Next.js) - Hosted on Vercel
2. **Backend API** (FastAPI) - Needs separate hosting

## Deployment Steps

### 1. Deploy Backend API

You need to deploy the backend API to a separate hosting service. Recommended options:

#### Option A: Render (Recommended)
1. Go to [render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repository
4. Select the `backend/` directory
5. Set the build command: `cd backend && pip install -r requirements.txt`
6. Set the start command: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
7. Deploy

#### Option B: Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Select the `backend/` directory
4. Railway will auto-detect the Python app
5. Deploy

#### Option C: Railway Alternative
1. Go to [fly.io](https://fly.io)
2. Install flyctl: `brew install flyctl`
3. Run: `fly launch` in the backend directory
4. Follow the prompts to deploy

### 2. Configure Backend URL

Once your backend is deployed, you'll get a URL like:
- `https://your-app.onrender.com`
- `https://your-app.up.railway.app`
- `https://your-app.fly.dev`

### 3. Configure Frontend for Vercel

#### Option A: Environment Variable (Recommended)
1. Go to your Vercel project settings
2. Add environment variable:
   - Key: `NEXT_PUBLIC_BACKEND_URL`
   - Value: `https://your-backend-url.com` (without /api)

#### Option B: Update Code
If you prefer to hardcode the URL, update `frontend/src/lib/api.ts`:
```javascript
const backendUrl = "https://your-backend-url.com";
```

### 4. Deploy Frontend to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Set the build directory to `frontend/`
4. Add the environment variable `NEXT_PUBLIC_BACKEND_URL`
5. Deploy

## Environment Variables

### Backend (.env)
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Frontend (Vercel Environment Variables)
```env
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.com
```

## Testing

After deployment:

1. **Test Backend**: Visit `https://your-backend-url.com/health`
2. **Test Frontend**: Visit your Vercel deployment URL
3. **Test Connection**: Try using the chat feature

## Troubleshooting

### Backend Issues
- Check logs in your backend hosting service
- Ensure `OPENAI_API_KEY` is set correctly
- Verify the API is accessible at `/health`

### Frontend Issues
- Check browser console for CORS errors
- Verify `NEXT_PUBLIC_BACKEND_URL` is set correctly
- Ensure the backend URL is accessible from the frontend

### CORS Issues
If you get CORS errors, add your frontend URL to the backend CORS settings in `backend/app/main.py`:
```python
origins = [
    "https://your-frontend.vercel.app",
    "http://localhost:3000"
]
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Browser  │───▶│   Vercel (Frontend)  │───▶│ Backend API     │
│                 │    │                    │    │ (Render/Railway)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Cost Considerations

- **Frontend on Vercel**: Free tier usually sufficient
- **Backend on Render**: Free tier with limitations, $7/month for basic
- **Backend on Railway**: Free tier available, $5/month for basic
- **OpenAI API**: Pay-per-use, costs depend on usage

## Production Checklist

- [ ] Backend deployed and accessible
- [ ] Environment variables configured
- [ ] CORS settings configured
- [ ] SSL certificates active
- [ ] Database configured (if using persistent storage)
- [ ] Monitoring and logging set up