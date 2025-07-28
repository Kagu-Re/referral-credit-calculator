# Deployment Guide - Referral Credit Calculator

## 🚀 Quick Deploy Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Referral Credit Calculator"
   git branch -M main
   git remote add origin https://github.com/yourusername/loyalty-calculator.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `yourusername/loyalty-calculator`
   - Main file path: `referral_calculator.py`
   - Click "Deploy!"

   **✅ Free, easy, and perfect for this app!**

### Option 2: Heroku

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app:**
   ```bash
   heroku create your-loyalty-calculator
   ```

3. **Deploy:**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Option 3: Railway

1. **Connect GitHub** at [railway.app](https://railway.app)
2. **Select repository** and deploy
3. **Set start command:** `streamlit run referral_calculator.py --server.port $PORT`

### Option 4: Local Network Sharing

**Quick local deployment for team testing:**
```bash
streamlit run referral_calculator.py --server.address 0.0.0.0 --server.port 8501
```
Then share your local IP: `http://YOUR_IP:8501`

## 📋 Pre-deployment Checklist

- ✅ `requirements.txt` created
- ✅ `.gitignore` created  
- ✅ Streamlit config created
- ✅ App tested locally
- ✅ No sensitive data in code

## 🔧 Environment Setup

If you need environment variables, create `.streamlit/secrets.toml`:
```toml
# Example secrets (don't commit this file)
API_KEY = "your-secret-key"
DATABASE_URL = "your-db-url"
```

## 🌐 Custom Domain (Optional)

After deploying, you can:
1. **Streamlit Cloud:** Use custom domain in settings
2. **Heroku:** Add custom domain in dashboard
3. **Railway:** Configure custom domain in project settings

## 📊 Usage Analytics (Optional)

Add to your Streamlit app:
```python
# Add to top of referral_calculator.py
st.markdown("""
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
window.dataLayer = window.dataLayer || [];
function gtag(){dataLayer.push(arguments);}
gtag('js', new Date());
gtag('config', 'GA_TRACKING_ID');
</script>
""", unsafe_allow_html=True)
```

## 🚨 Security Notes

- Never commit secrets to GitHub
- Use environment variables for sensitive data
- Consider adding authentication for internal tools
- Regularly update dependencies

## 📱 Mobile Optimization

Your app is already mobile-friendly with Streamlit's responsive design!

---

**Recommended: Start with Streamlit Community Cloud - it's free and perfect for this calculator!**
