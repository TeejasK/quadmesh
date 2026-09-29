# 🚀 Quadmesh AI Web App — Vercel Free Hosting & Bank Setup Guide

This guide walks you through deploying the **Quadmesh AI Web App** for free on **Vercel**, configuring **Google Authentication**, connecting your **Indian Bank Account** via Razorpay for automatic subscription credits (INR 699, 1,899, 7,999), and utilizing the **Max-Size 3D CAD Studio** with dual **Blender + WebApp** generation.

---

## 🎨 1. Highlights of the Claude-Light Web App

- **Claude Minimalist Light Theme**: Warm cream palette (`#FBFBFA`), terracotta accents (`#C96A36`), refined borders (`#E5E3DC`), and crisp typography.
- **Maximized 3D CAD Studio Viewport**: The 3D Three.js canvas occupies 100% of the screen.
- **Collapsible / Dockable Side Panels**:
  - **Left Panel (CAD Planner & 9-Roles)**: Has **Minimize** (collapses to an edge pill) and **Maximize** (expands to wide view).
  - **Right Panel (CadQuery Code & FEA Audit)**: Has **Minimize** and **Maximize** toggles.
- **Dual Engine Generation (Blender + WebApp)**: 
  - Generates watertight 3D meshes right inside the WebApp browser viewport.
  - Concurrently streams real socket operations to your live Blender scene via the Quadmesh Python socket bridge (`/api/open_blender` & `/api/chat`).
- **Google Authentication**: Google Identity Services integration with user profile and session states.
- **INR Subscription Pricing with Direct Bank Settlement**:
  - **Quad-Plus**: ₹699 / month (100M & 500M Models, 250 exports/mo)
  - **Quad-Pro**: ₹1,899 / month (1B & 3B Flagship, Unlimited exports, Blender bridge, Nat Voice)
  - **Quad-Max**: ₹7,999 / month (7B Flagship, 50+ Component assemblies, Automated FEA, VLA automation)
  - **Quad-Enterprise**: Pay-As-You-Use (₹0.45 per operation, custom fine-tuning)

---

## ⚡ 2. Immediate Viewing (No Setup Needed)

You can open the complete standalone template immediately in any web browser:
1. Double-click or open:
   ```
   d:\quadmesh-project\quadmesh_project\quadmesh_app_template.html
   ```
2. Or start the Python FastAPI backend server:
   ```powershell
   cd D:\quadmesh-project\quadmesh_project
   .\.venv\Scripts\Activate.ps1
   python -m quadmesh.webapp.server
   ```
   Open `http://127.0.0.1:8420` in your browser.

---

## ☁️ 3. Deploying to Vercel for Free

The `web/` folder is pre-configured with `vercel.json` and Serverless API routes.

### Option A: Using Vercel CLI (Fastest — 2 Minutes)
1. Install the Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```
2. Navigate to the `web` folder:
   ```powershell
   cd D:\quadmesh-project\quadmesh_project\web
   ```
3. Run the deploy command:
   ```powershell
   vercel
   ```
4. Follow the short terminal prompts (Press Enter to accept defaults).
5. For production deployment:
   ```powershell
   vercel --prod
   ```
   Your app will be live on `https://quadmesh-ai.vercel.app` (or your chosen project name)!

---

### Option B: Deploying via GitHub & Vercel Dashboard
1. Push your repository to GitHub:
   ```powershell
   git add .
   git commit -m "Deploy Quadmesh Claude-style Web App"
   git push origin main
   ```
2. Go to [vercel.com](https://vercel.com) and sign in for free.
3. Click **"Add New Project"** -> Select your **`quadmesh_project`** repository.
4. In the configuration:
   - **Root Directory**: Select `web`
   - **Framework Preset**: Other / Static
5. Click **"Deploy"**.

---

## 💳 4. Connecting Your Indian Bank Account for Credits

All subscription payments from **Quad-Plus (₹699)**, **Quad-Pro (₹1,899)**, and **Quad-Max (₹7,999)** are processed via Razorpay in INR with direct bank payouts.

### Setting up Razorpay Payouts to Your Bank:
1. Create a free account at [razorpay.com](https://razorpay.com).
2. Go to **Settings** -> **API Keys** -> Generate your `Key ID` and `Key Secret`.
3. Go to **Banking & Settlements** -> **Add Bank Account**:
   - **Account Holder Name**: `Teejas K`
   - **Bank Name**: e.g., `HDFC Bank`
   - **Account Number**: e.g., `50100482910394`
   - **IFSC Code**: e.g., `HDFC0000240`
   - **UPI ID**: e.g., `quadmesh@hdfcbank`
4. In Vercel Project Settings -> **Environment Variables**, add:
   - `RAZORPAY_KEY_ID` = `rzp_live_your_key_id`
   - `RAZORPAY_KEY_SECRET` = `your_secret_key`
5. Payouts from user transactions will automatically be credited to your bank on a **T+1** daily settlement schedule.

---

## 🔑 5. Setting up Google Authentication

1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials).
2. Create an **OAuth 2.0 Client ID** (Web Application).
3. Add Authorized JavaScript Origins:
   - `http://localhost:8420`
   - `https://your-quadmesh-app.vercel.app`
4. Copy your **Client ID** and set it in your environment or template.

---

## 🛠️ 6. Running Dual Blender + WebApp CAD Generation

When running locally:
1. Start the Blender bridge by clicking **"Blender: Connected"** in the top navigation bar or running:
   ```powershell
   python -m quadmesh.webapp.server
   ```
2. Type any CAD prompt, e.g.:
   - *"Involute spur gear module 2 with 20 teeth, center bore, and keyway"*
   - *"Hexacopter drone motor mount arm with M3 bolt patterns"*
   - *"Robotic clevis elbow joint with bearing recesses"*
3. Watch the geometry render instantly in the **Max-Size 3D Viewport**, while Blender constructs the identical shape live!
