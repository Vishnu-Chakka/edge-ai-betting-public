#!/bin/bash

# Vercel Configuration Script
# Run this AFTER deploying your backend to Render/Railway

echo "🔧 Vercel Configuration Helper"
echo "================================"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "📋 This script will help you configure Vercel environment variables."
echo ""
echo "First, deploy your backend to Render using the guide in DEPLOY_NOW.md"
echo ""

# Prompt for backend URL
read -p "Enter your backend URL (e.g., https://edge-ai-betting-backend.onrender.com): " BACKEND_URL

if [ -z "$BACKEND_URL" ]; then
    echo "❌ Backend URL is required. Exiting."
    exit 1
fi

# Trim trailing slash if present
BACKEND_URL=${BACKEND_URL%/}

echo ""
echo "Setting up environment variables in Vercel..."
echo ""

# Add NEXT_PUBLIC_API_URL
echo "Adding NEXT_PUBLIC_API_URL..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_API_URL production

echo "Adding NEXT_PUBLIC_API_URL to preview..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_API_URL preview

echo "Adding NEXT_PUBLIC_API_URL to development..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_API_URL development

# Add NEXT_PUBLIC_BACKEND_URL (for compatibility)
echo ""
echo "Adding NEXT_PUBLIC_BACKEND_URL..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_BACKEND_URL production

echo "Adding NEXT_PUBLIC_BACKEND_URL to preview..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_BACKEND_URL preview

echo "Adding NEXT_PUBLIC_BACKEND_URL to development..."
echo "$BACKEND_URL" | vercel env add NEXT_PUBLIC_BACKEND_URL development

echo ""
echo "✅ Environment variables added!"
echo ""
echo "🚀 Now redeploying to Vercel..."
echo ""

# Redeploy
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Your site should now work exactly like localhost:"
echo "   - Frontend: https://edge-ai-betting-public.vercel.app"
echo "   - Backend: $BACKEND_URL"
echo ""
echo "🧪 Test your deployment:"
echo "   1. Visit your Vercel URL"
echo "   2. Click 'START CHATTING NOW'"
echo "   3. Try sending a message"
echo ""
echo "Done! 🎉"
