# 🤖 Introduction to LLM - Practice Assignment Quiz

An interactive Streamlit web application for practicing assignment questions from the "Introduction to Large Language Models (LLMs)" course. This quiz application allows students to test their knowledge across all 12 weeks of course material.

## 🚀 Features

- **Multi-Week Selection**: Choose from any combination of weeks 1-12
- **Question Types Support**: 
  - Multiple Choice Questions (MCQ)
  - Multiple Select Questions (MSQ) 
  - Numeric Questions
- **Interactive Quiz Interface**: Clean, modern UI with real-time scoring
- **Shuffle Options**: 
  - Shuffle questions within selected weeks
  - Shuffle answer options for each question
- **Detailed Results**: Comprehensive scoring with points, percentages, and grades
- **Responsive Design**: Works on desktop and mobile devices

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd introduction-to-llm-quiz
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501` to access the quiz application.

## 📁 Project Structure

```
introduction_to_llm/
├── streamlit_app.py          # Main Streamlit application
├── data/
│   ├── All_Weeks_Questions.csv    # CSV data source
│   └── all_weeks_assignments.json # JSON data source (preferred)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🎯 How to Use

1. **Select Weeks**: Choose one or more weeks from the available options
2. **Configure Quiz**: 
   - Enable "Shuffle Questions" to randomize question order
   - Enable "Shuffle Options" to randomize answer choices
3. **Answer Questions**: 
   - For MCQ: Select one answer
   - For MSQ: Select multiple answers
   - For Numeric: Enter a number
4. **Submit Quiz**: Click "Submit Quiz" to see your results
5. **Review Results**: View detailed scoring, correct answers, and explanations

## 📊 Data Sources

The application supports two data formats:

- **JSON Format** (Preferred): `data/all_weeks_assignments.json`
- **CSV Format**: `data/All_Weeks_Questions.csv`

The JSON format is automatically used if both files are present, as it provides better structure and metadata.

## 🎨 Customization

### Styling
The application uses custom CSS for enhanced visual appeal. Key styling elements include:
- Gradient header with course branding
- Compact week selection cards
- Clean question presentation with separators
- Color-coded result indicators

### Data Format
To add new questions or modify existing ones, update the JSON file following this structure:

```json
{
  "course": "Introduction to Large Language Models (LLMs)",
  "total_weeks": 12,
  "assignments": [
    {
      "week": 1,
      "title": "Week 1 : Assignment 1",
      "questions": [
        {
          "question_id": 1,
          "points": 1,
          "question_text": "Your question here?",
          "question_type": "MCQ",
          "options": ["Option A", "Option B", "Option C", "Option D"],
          "correct_answer": "Option A"
        }
      ]
    }
  ]
}
```

## 🔧 Technical Details

- **Framework**: Streamlit
- **Data Processing**: JSON/CSV parsing with pandas
- **Session Management**: Streamlit session state for persistent UI
- **Responsive Design**: CSS Grid and Flexbox layouts
- **Error Handling**: Robust error handling for data loading and user input

## 📝 Question Types

### Multiple Choice Questions (MCQ)
- Single correct answer
- Radio button interface
- Points: 1-2 per question

### Multiple Select Questions (MSQ)
- Multiple correct answers
- Checkbox interface
- Partial credit for correct selections

### Numeric Questions
- Exact numeric answers
- Text input interface
- String/numeric comparison for scoring

## 🎓 Grading System

- **Points**: Each question has assigned point values
- **Percentage**: Calculated as (earned points / total points) × 100
- **Grade**: A+ (90-100%), A (80-89%), B (70-79%), C (60-69%), D (50-59%), F (<50%)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is for educational purposes. All course content credits belong to NPTEL.

## 🙏 Acknowledgments

- **Course Content**: NPTEL Introduction to Large Language Models (LLMs)
- **Course Link**: https://onlinecourses.nptel.ac.in/noc25_cs161/preview
- **Framework**: Streamlit for web application development

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Data loading errors**
   - Ensure `data/all_weeks_assignments.json` exists
   - Check JSON format validity

3. **Module not found errors**
   ```bash
   pip install -r requirements.txt
   ```

### Support

For issues and questions, please create an issue in the repository or contact the maintainers.

---

**Note**: This application is designed for educational purposes. We do not own any of the content on this website. All credits to NPTEL.