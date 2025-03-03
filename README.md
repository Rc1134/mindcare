# MindCare

MindCare is an AI-powered mental health support platform designed to assist users with mindfulness, mood tracking, meditation, planning, and AI-driven mental health guidance.

## Features

### 1. **AI Chatbot**
- Provides mental health support and guidance.
- Trained on a CSV file using Hugging Face embeddings.
- Uses the Mistral 7B LLM via the Hugging Face API.

### 2. **Planner & Organizer**
- Helps users plan and record tasks.
- AI-powered prioritization for better task management.

### 3. **Journal**
- Allows users to write and maintain previous journal records.
- AI-based insights on journal entries.

### 4. **Authentication**
- User registration and login functionality.

### 5. **Resources Page**
- Provides books, exercises, and helpful materials.

### 6. **Meditation Corner**
- Stopwatch-style meditation timer (1, 3, 5, 10 min).
- Tracks previous meditation sessions.
- Plays relaxing music based on mood.

### 7. **Mood Tracking & Sentiment Analysis**
- Recognizes user mood trends.
- Suggests coping strategies and relaxation techniques.
- Suggests songs to improve mood



**Made using streamlit and langchain**

## Installation & Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/Rc1134/mindcare.git
   cd mindcare
   ```

2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Run the application:
   ```sh
   flask run
   ```

5. Open your browser and visit `http://127.0.0.1:5000/`

## Contributing
Contributions are welcome! Feel free to open issues and submit pull requests.

## License
This project is licensed under the MIT License.

