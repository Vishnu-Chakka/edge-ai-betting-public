#!/bin/bash

# Edge AI Final - Vercel Deployment Script

echo "🚀 Deploying edge-ai-final to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Please install it first:"
    echo "   npm install -g vercel"
    exit 1
fi

# Login to Vercel (if not already logged in)
echo "🔐 Checking Vercel authentication..."
vercel login

# Navigate to project root
cd "$(dirname "$0")"

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to your Vercel dashboard"
echo "2. Select your project"
echo "3. Go to Settings → Environment Variables"
echo "4. Add these variables:"
echo "   - NEXT_PUBLIC_BACKEND_URL: [your backend URL from Render/Railway]"
echo "   - ANTHROPIC_API_KEY: [your Anthropic API key]"
echo "   - ODDS_API_KEY: [your odds API key]"
echo ""
echo "5. Redeploy your project"
echo ""
echo "🔗 Your frontend will be available at: https://edge-ai-final.vercel.app"