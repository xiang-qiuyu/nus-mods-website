# NUSMods Timetable Optimizer

AI-powered timetable scheduler using Google OR-Tools and NUSMods API

## 🚀 Features

- **Intelligent Scheduling**: Uses Google OR-Tools constraint solver
- **Multiple Preferences**: No morning classes, free Fridays, lunch breaks, compact schedule, minimize travel
- **Multiple Solutions**: Generates up to 5 optimized timetables
- **NUSMods Integration**: Direct export to NUSMods and .ics calendar
- **Trade-off Explanations**: Clear explanations of scheduling decisions

---

## 📋 Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 14+** (for frontend)
- **pip** (Python package manager)
- **npm** (Node package manager)

---

## 🛠️ Installation & Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/xiang-qiuyu/nus-mods-website.git
cd nus-mods-website
```

### **2. Backend Setup (Python)**

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **3. Frontend Setup (React)**

```bash
# Navigate back to project root
cd ..

# Install frontend dependencies
npm install
```

---

## ▶️ Running the Application

You need to run **both** backend and frontend servers:

### **Terminal 1 - Start Backend**

```bash
cd backend
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
python backend.py
```

Backend will run on: `http://localhost:5000`

### **Terminal 2 - Start Frontend**

```bash
# In project root
npm run dev
```

Frontend will run on: `http://localhost:5173`

### **3. Open in Browser**

Navigate to: `http://localhost:5173`

---

## 📦 Dependencies

### **Backend (Python)**
- Flask - Web framework
- flask-cors - Cross-origin resource sharing
- ortools - Google's optimization library
- requests - HTTP library

### **Frontend (React)**
- React 18
- Vite - Build tool
- lucide-react - Icons

---

## 🎯 Usage

1. Enter your NUS module codes (comma-separated)
   - Example: `CS2030S, CS2040S, MA1521`

2. Select your preferences:
   - ☑️ No classes before 10am
   - ☑️ Compact schedule (minimize gaps)
   - ☑️ Keep Fridays free
   - ☑️ Daily lunch break (12-2pm)
   - ☑️ Minimize campus travel

3. Click **"Generate Optimal Timetables"**

4. Review generated timetables with scores and trade-offs

5. Export to:
   - **NUSMods** - Direct link to view in NUSMods
   - **.ics file** - Import to Google Calendar/Outlook

---

## 🏗️ Project Structure

```
nus-mods-optimizer/
├── backend/
│   ├── venv/              # Virtual environment
│   ├── backend.py         # Flask API + OR-Tools optimizer
│   └── requirements.txt   # Python dependencies
├── src/
│   ├── App.jsx           # Main React component
│   ├── App.css           # Styles
│   └── main.jsx          # Entry point
├── package.json          # Node dependencies
├── vite.config.js        # Vite configuration
└── README.md             # This file
```

---

## 🔧 API Endpoints

### `GET /api/health`
Health check endpoint

### `POST /api/optimize`
Generate optimized timetables

**Request Body:**
```json
{
  "modules": ["CS2030S", "CS2040S", "MA1521"],
  "preferences": {
    "noMorningClasses": true,
    "compactSchedule": false,
    "freeFridays": true,
    "lunchBreak": true,
    "minimizeTravel": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "timetables": [...]
}
```

### `GET /api/modules/{module_code}`
Get information about a specific module

---

## 🧪 Testing

Test the backend API directly:

```bash
# Health check
curl http://localhost:5000/api/health

# Get module info
curl http://localhost:5000/api/modules/CS2030S
```

---

## 🐛 Troubleshooting

### **Backend won't start**
- Make sure virtual environment is activated
- Check if all dependencies are installed: `pip list`
- Verify Python version: `python --version` (should be 3.8+)

### **Frontend won't start**
- Delete `node_modules/` and run `npm install` again
- Check Node version: `node --version` (should be 14+)

### **CORS errors**
- Ensure backend is running on port 5000
- Check that `flask-cors` is installed

### **No timetables generated**
- Verify module codes are correct and available for current semester
- Try with fewer preferences enabled
- Check backend terminal for error messages

---

## 👥 Team Members

- Chen Tianwei
- Chua Yong Liang
- Ron Quah Kai Yi
- Xiang Qiuyu

---

## 📄 License

This project is for educational purposes as part of NUS coursework.

---

## 🙏 Acknowledgments

- **NUSMods API** - Module data source
- **Google OR-Tools** - Constraint optimization solver
- **Flask** - Backend framework
- **React** - Frontend framework