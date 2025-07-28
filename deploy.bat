@echo off
echo 🚀 Referral Credit Calculator - Public Deployment
echo ================================================
echo.
echo Your app is ready for public deployment!
echo.
echo STEP 1: Create GitHub Repository
echo --------------------------------
echo 1. Go to https://github.com/new
echo 2. Repository name: referral-credit-calculator
echo 3. Make it PUBLIC (required for free Streamlit hosting)
echo 4. Don't initialize with README
echo 5. Click "Create repository"
echo.
echo STEP 2: Push Your Code
echo ---------------------
echo Copy and run these commands:
echo.
echo git remote add origin https://github.com/YOUR_USERNAME/referral-credit-calculator.git
echo git branch -M main  
echo git push -u origin main
echo.
echo STEP 3: Deploy on Streamlit Cloud
echo ---------------------------------
echo 1. Visit https://share.streamlit.io
echo 2. Sign in with GitHub
echo 3. Click "New app"
echo 4. Select your repository: YOUR_USERNAME/referral-credit-calculator
echo 5. Main file path: referral_calculator.py
echo 6. Click "Deploy!"
echo.
echo 🎉 Your app will be live in ~2-3 minutes!
echo 📱 The URL will be: https://YOUR_USERNAME-referral-credit-calculator-main-referral-calc-xyz123.streamlit.app
echo.
pause
