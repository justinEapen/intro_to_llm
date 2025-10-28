# Introduction to LLM Quiz - Deployment Guide

## 🚀 Quick Deploy to Streamlit Cloud

### Step 1: Upload Files to GitHub
Upload these files to your repository `https://github.com/justinEapen/intro_to_llm`:

**Required Files:**
- `streamlit_app.py` (main application)
- `requirements.txt` (dependencies)
- `README.md` (documentation)
- `data/all_weeks_assignments.json` (quiz data)

**File Structure:**
```
intro_to_llm/
├── streamlit_app.py
├── requirements.txt
├── README.md
└── data/
    └── all_weeks_assignments.json
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `justinEapen/intro_to_llm`
5. Set main file path: `streamlit_app.py`
6. Click "Deploy!"

### Step 3: Configure App Settings
- **App URL**: Will be auto-generated (e.g., `https://intro-to-llm-quiz.streamlit.app`)
- **Repository**: `justinEapen/intro_to_llm`
- **Branch**: `main`
- **Main file path**: `streamlit_app.py`

## 📁 Files to Upload

### 1. streamlit_app.py
The main Streamlit application file (already created)

### 2. requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
```

### 3. README.md
Comprehensive documentation (already created)

### 4. data/all_weeks_assignments.json
The quiz data file (already exists)

## 🔧 Local Testing Before Deploy

Test locally first:
```bash
cd introduction_to_llm
streamlit run streamlit_app.py
```

## 🌐 Post-Deployment

After deployment:
1. Your app will be live at `https://[app-name].streamlit.app`
2. Share the URL with students
3. Monitor usage in Streamlit Cloud dashboard

## 📝 Notes

- Streamlit Cloud automatically detects `requirements.txt`
- The app will restart automatically when you push changes to GitHub
- All data is loaded from the JSON file in the `data/` folder
- No database setup required - everything runs in memory

## 🆘 Troubleshooting

**If deployment fails:**
1. Check that all files are in the correct structure
2. Verify `requirements.txt` syntax
3. Ensure `streamlit_app.py` runs locally without errors
4. Check Streamlit Cloud logs for specific error messages

**Common issues:**
- Missing `data/` folder or JSON file
- Incorrect file paths in the code
- Missing dependencies in `requirements.txt`
