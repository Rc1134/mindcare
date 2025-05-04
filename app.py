import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import time
import random
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
print("Hugging Face Token:", hf_token)

# Initialize session state variables if they don't exist
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'journals' not in st.session_state:
    st.session_state.journals = {}
if 'plans' not in st.session_state:
    st.session_state.plans = {}
if 'meditation_logs' not in st.session_state:
    st.session_state.meditation_logs = {}
if 'login_attempts' not in st.session_state:
    st.session_state.login_attempts = 0
if 'check_in_streak' not in st.session_state:
    st.session_state.check_in_streak = {}
if 'last_check_in' not in st.session_state:
    st.session_state.last_check_in = {}

# Load data function for the chatbot
@st.cache_resource
def load_chatbot_data():
    try:
        # Load data from the specified CSV file
        df = pd.read_csv('Mental_Health_Faq.csv')
        
        # Check if required columns exist
        required_cols = ['Questions', 'Answers']
        if not all(col in df.columns for col in required_cols):
            st.error("CSV file must contain 'question' and 'answer' columns")
            # Provide fallback data in case CSV is improperly formatted
            fallback_data = {
                'question': [
                    'How can I manage stress?',
                    'What are some relaxation techniques?',
                    'How to improve sleep?'
                ],
                'answer': [
                    'Manage stress through regular exercise, adequate sleep, deep breathing, and time management.',
                    'Relaxation techniques include deep breathing, progressive muscle relaxation, meditation, and visualization.',
                    'Improve sleep by maintaining a regular schedule, creating a restful environment, limiting screen time before bed, and avoiding caffeine.'
                ]
            }
            return pd.DataFrame(fallback_data)
        
        return df
    except FileNotFoundError:
        st.error("mental_health_faq.csv file not found. Please ensure the file exists in the application directory.")
        # Create a fallback dataset if CSV isn't found
        fallback_data = {
            'question': [
                'How can I manage stress?',
                'What are some relaxation techniques?',
                'How to improve sleep?',
                'How to start meditation?',
                'What to do when feeling anxious?',
                'How to practice mindfulness?',
                'What are signs of depression?',
                'How to build healthy habits?',
                'How to stop negative thinking?',
                'What is cognitive behavioral therapy?'
            ],
            'answer': [
                'Manage stress through regular exercise, adequate sleep, deep breathing, and time management.',
                'Relaxation techniques include deep breathing, progressive muscle relaxation, meditation, and visualization.',
                'Improve sleep by maintaining a regular schedule, creating a restful environment, limiting screen time before bed, and avoiding caffeine.',
                'Start meditation with just 5 minutes daily in a quiet space, focusing on your breath. Gradually increase duration as you get comfortable.',
                'When feeling anxious, try deep breathing, grounding techniques, physical exercise, or talking to someone you trust.',
                'Practice mindfulness by paying attention to the present moment without judgment, through meditation, mindful eating, or body scan exercises.',
                'Signs of depression include persistent sadness, loss of interest, changes in sleep or appetite, fatigue, and feelings of worthlessness.',
                'Build healthy habits by starting small, being consistent, tracking progress, and celebrating small victories.',
                'Combat negative thinking by challenging irrational thoughts, practicing gratitude, and focusing on solutions instead of problems.',
                'Cognitive behavioral therapy (CBT) is a type of psychological treatment that helps people identify and change negative thought patterns.'
            ]
        }
        return pd.DataFrame(fallback_data)
    except Exception as e:
        st.error(f"Error loading chatbot data: {e}")
        return pd.DataFrame(columns=['Questions', 'Answers'])

# Load embedding model
@st.cache_resource
def load_embedding_model():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Initialize Hugging Face client
@st.cache_resource
def init_hf_client():
    try:
        # Replace with your API token in production
        client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token
)

        return client
    except Exception as e:
        st.error(f"Error initializing Hugging Face client: {e}")
        return None

# Function to get embeddings
def get_embeddings(model, texts):
    return model.encode(texts)

def get_hf_response(client, question):
    try:
        if client is None:
            return "Model service is currently unavailable."
        
        prompt = f"""<s>[INST] You are a helpful mental health assistant. Provide a brief, supportive response to this question in few words or a sentence and based on chat history:
        
        {question} [/INST]</s>"""
        
        response = client.text_generation(
            prompt, 
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Save and load functions
def save_data():
    data = {
        'users': st.session_state.users,
        'journals': st.session_state.journals,
        'plans': st.session_state.plans,
        'meditation_logs': st.session_state.meditation_logs,
        'check_in_streak': st.session_state.check_in_streak,
        'last_check_in': st.session_state.last_check_in
    }
    with open('app_data.json', 'w') as f:
        json.dump(data, f)

def load_data():
    if os.path.exists('app_data.json'):
        with open('app_data.json', 'r') as f:
            data = json.load(f)
            st.session_state.users = data.get('users', {})
            st.session_state.journals = data.get('journals', {})
            st.session_state.plans = data.get('plans', {})
            st.session_state.meditation_logs = data.get('meditation_logs', {})
            st.session_state.check_in_streak = data.get('check_in_streak', {})
            st.session_state.last_check_in = data.get('last_check_in', {})

# Authentication functions
def login():
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username]['password'] == password:
            st.session_state.authenticated = True
            st.session_state.current_user = username
            update_streak(username)
            st.success("Successfully logged in!")
            st.rerun()
        else:
            st.session_state.login_attempts += 1
            st.error("Invalid username or password")

def register():
    st.header("Register")
    new_username = st.text_input("Choose Username")
    new_password = st.text_input("Choose Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    email = st.text_input("Email")
    
    if st.button("Register"):
        if new_username in st.session_state.users:
            st.error("Username already exists")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        elif not new_username or not new_password or not email:
            st.error("All fields are required")
        else:
            st.session_state.users[new_username] = {
                'password': new_password,
                'email': email,
                'joined_date': datetime.datetime.now().strftime("%Y-%m-%d")
            }
            st.session_state.journals[new_username] = []
            st.session_state.plans[new_username] = []
            st.session_state.meditation_logs[new_username] = []
            st.session_state.check_in_streak[new_username] = 1
            st.session_state.last_check_in[new_username] = datetime.datetime.now().strftime("%Y-%m-%d")
            save_data()
            st.success("Registration successful! Please log in.")

def logout():
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.rerun()

def update_streak(username):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if username not in st.session_state.last_check_in:
        st.session_state.last_check_in[username] = today
        st.session_state.check_in_streak[username] = 1
    else:
        last_date = datetime.datetime.strptime(st.session_state.last_check_in[username], "%Y-%m-%d")
        current_date = datetime.datetime.strptime(today, "%Y-%m-%d")
        delta = (current_date - last_date).days
        
        if delta == 1:  # consecutive day
            st.session_state.check_in_streak[username] += 1
        elif delta > 1:  # streak broken
            st.session_state.check_in_streak[username] = 1
        # if same day, no change
        
        st.session_state.last_check_in[username] = today
    save_data()

import time
import streamlit as st

def ai_assistant_page():
    st.header("🤖 AI Mental Health Assistant")
    
    # Initialize chat history and processing state in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
    
    # Function to handle user input submission
    def handle_input(user_question):
        if user_question:
            # Add user question to the chat history
            st.session_state.chat_history.append(("👤 You", user_question))
            # Set processing flag to trigger response generation
            st.session_state.processing = True
    
    # Display the chat history in a chat-like format
    for sender, message in st.session_state.chat_history:
        if sender == "👤 You":
            # User's message
            with st.chat_message(sender, avatar="👤"):
                st.markdown(f"<div style='text-align:left; color: #1E1E1E; background-color: #D1E7FF; border-radius: 10px; padding: 10px;'>{message}</div>", unsafe_allow_html=True)
        else:
            # AI's message
            with st.chat_message(sender, avatar="🤖"):
                st.markdown(f"<div style='text-align: left; color: #1E1E1E; background-color: #F1F1F1; border-radius: 10px; padding: 10px;'>{message}</div>", unsafe_allow_html=True)
    
    # Process response if the processing flag is set
    if st.session_state.processing:
        # Get the last user message
        last_user_message = st.session_state.chat_history[-1][1]
        
        # Generate response with a spinner
        with st.spinner("Generating detailed response..."):
            response = get_hf_response(init_hf_client(), last_user_message)
        
        # Add AI response to the chat history
        st.session_state.chat_history.append(("🤖 AI", response))
        
        # Reset the processing flag
        st.session_state.processing = False
        
        # Force a rerun to display the response
        st.rerun()
    
    # Clear chat history button
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.processing = False
    # Input field for user message - using st.chat_input
    user_question = st.chat_input("Ask me anything about mental health:")
    if user_question:
        handle_input(user_question)
        st.rerun()  # Only rerun once to show the user message




def journal_page():
    st.header("My Journal")
    
    username = st.session_state.current_user
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Create a new journal entry
    st.subheader("New Entry")
    entry_date = st.date_input("Date", datetime.datetime.now())
    mood = st.select_slider("How are you feeling today?", options=["Very Bad", "Bad", "Neutral", "Good", "Very Good"])
    entry_title = st.text_input("Title")
    entry_content = st.text_area("Write your thoughts...", height=200)
    
    if st.button("Save Entry"):
        if entry_title and entry_content:
            new_entry = {
                'date': entry_date.strftime("%Y-%m-%d"),
                'title': entry_title,
                'content': entry_content,
                'mood': mood
            }
            if username not in st.session_state.journals:
                st.session_state.journals[username] = []
            st.session_state.journals[username].append(new_entry)
            save_data()
            st.success("Journal entry saved!")
        else:
            st.error("Title and content are required")
    
    # View previous entries
    st.subheader("Previous Entries")
    if username in st.session_state.journals and st.session_state.journals[username]:
        entries = st.session_state.journals[username]
        entries.sort(key=lambda x: x['date'], reverse=True)
        
        for i, entry in enumerate(entries):
            with st.expander(f"{entry['date']} - {entry['title']} ({entry['mood']})"):
                st.write(entry['content'])
                if st.button(f"Delete Entry {i}"):
                    st.session_state.journals[username].remove(entry)
                    save_data()
                    st.rerun()

    else:
        st.info("No journal entries yet. Start writing today!")
    st.subheader("📊 Mood Insights")

    if st.session_state.current_user in st.session_state.journals and st.session_state.journals[st.session_state.current_user]:
        journal_entries = st.session_state.journals[st.session_state.current_user]

        # Extract dates and moods
        dates = [entry['date'] for entry in journal_entries]
        moods = [entry['mood'] for entry in journal_entries]

        # Convert mood labels to numeric values for visualization
        mood_map = {"Very Bad": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Very Good": 5}
        mood_values = [mood_map[mood] for mood in moods]

        # Display a line chart for mood trends
        mood_df = pd.DataFrame({"Date": dates, "Mood Score": mood_values})
        mood_df["Date"] = pd.to_datetime(mood_df["Date"])
        mood_df = mood_df.sort_values("Date")

        st.line_chart(mood_df.set_index("Date"))

        # Provide insights
        if len(moods) > 3:
            last_three_moods = moods[-3:]
            if last_three_moods.count("Bad") > 1 or last_three_moods.count("Very Bad") > 1:
                st.warning("It looks like you've been feeling down lately. Consider trying meditation or talking to someone you trust.")
            elif last_three_moods.count("Good") > 1 or last_three_moods.count("Very Good") > 1:
                st.success("You've been feeling great! Keep up with your healthy habits.")

    else:
        st.info("Write journal entries to track your mood over time.")
        

    

def get_music_recommendation(mood):
    """
    Selects a random downloaded music file based on the user's mood.
    """
    mood_music = {
        "Very Good": ["music/very_good_1.mp3"],
        "Good": ["music/good_1.mp3"],
        "Neutral": ["music/neutral_1.mp3"],
        "Bad": ["music/bad_1.mp3"],
        "Very Bad": ["music/very_bad_1.mp3"]
    }

    if mood in mood_music and mood_music[mood]:
        return random.choice(mood_music[mood])
    
    return None  # No music available




def analyze_mood(username):
    """
    Analyze the user's mood based on their last few journal entries.
    Returns a mood summary and a recommended self-care action.
    """
    if username not in st.session_state.journals or not st.session_state.journals[username]:
        return "No journal entries yet.", "Try writing about your day to track your mood."

    # Get last 5 journal entries
    last_entries = st.session_state.journals[username][-5:]
    moods = [entry["mood"] for entry in last_entries]

    # Count occurrences of each mood
    mood_counts = {mood: moods.count(mood) for mood in set(moods)}
    
    # Determine the most frequent mood
    most_common_mood = max(mood_counts, key=mood_counts.get)

    # Recommend an action based on mood
    recommendations = {
        "Very Good": "Keep up the great work! Try spreading positivity today.",
        "Good": "You’re doing well! Consider a short meditation session.",
        "Neutral": "Maybe take a walk or listen to calming music.",
        "Bad": "Try writing down things you're grateful for.",
        "Very Bad": "Consider talking to a friend or practicing deep breathing."
    }
    return f"Your most frequent mood is **{most_common_mood}**.", recommendations[most_common_mood]

def generate_daily_challenge():
    """
    Generates a daily self-care challenge based on user's past moods.
    """
    challenges = {
        "Very Good": ["Call a loved one and share your happiness.", "Try a new hobby today!", "Write 3 things you're grateful for."],
        "Good": ["Go for a short walk outdoors.", "Listen to calming music for 10 minutes.", "Practice 5 minutes of deep breathing."],
        "Neutral": ["Write down how you feel in a journal.", "Watch a motivational video.", "Do a quick stretching session."],
        "Bad": ["Try progressive muscle relaxation.", "Write down a small goal for today.", "Drink water and take deep breaths."],
        "Very Bad": ["Reach out to a friend or therapist.", "Practice 10 minutes of guided meditation.", "Write 5 positive affirmations."]
    }

    username = st.session_state.current_user
    if username:
        mood_summary, _ = analyze_mood(username)
        mood = mood_summary.split("**")[1] if "**" in mood_summary else "Neutral"
        return random.choice(challenges.get(mood, challenges["Neutral"]))

    return "Take a moment to check in with yourself today."




def planner_page():
    st.header("My Planner")
    
    username = st.session_state.current_user
    
    # Create a new plan
    st.subheader("Add New Task/Plan")
    task_date = st.date_input("Date", datetime.datetime.now())
    task_title = st.text_input("Task Title")
    task_description = st.text_area("Description")
    task_priority = st.select_slider("Priority", options=["Low", "Medium", "High"])
    
    if st.button("Add Task"):
        if task_title:
            new_task = {
                'date': task_date.strftime("%Y-%m-%d"),
                'title': task_title,
                'description': task_description,
                'priority': task_priority,
                'completed': False
            }
            if username not in st.session_state.plans:
                st.session_state.plans[username] = []
            st.session_state.plans[username].append(new_task)
            save_data()
            st.success("Task added!")
        else:
            st.error("Task title is required")
    
    # View and manage tasks
    st.subheader("My Tasks")
    if username in st.session_state.plans and st.session_state.plans[username]:
        # Filter options
        filter_option = st.radio("Filter", ["All", "Upcoming", "Completed", "By Priority"])
        
        tasks = st.session_state.plans[username]
        filtered_tasks = tasks
        
        if filter_option == "Upcoming":
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            filtered_tasks = [t for t in tasks if t['date'] >= today and not t['completed']]
        elif filter_option == "Completed":
            filtered_tasks = [t for t in tasks if t['completed']]
        elif filter_option == "By Priority":
            priority = st.selectbox("Select Priority", ["High", "Medium", "Low"])
            filtered_tasks = [t for t in tasks if t['priority'] == priority]
        
        filtered_tasks.sort(key=lambda x: x['date'])
        
        for i, task in enumerate(filtered_tasks):
            col1, col2 = st.columns([4, 1])
            with col1:
                status = "✅" if task['completed'] else "⏳"
                priority_color = {
                    "High": "🔴",
                    "Medium": "🟠",
                    "Low": "🟢"
                }
                with st.expander(f"{task['date']} - {status} {priority_color[task['priority']]} {task['title']}"):
                    st.write(task['description'])
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button(f"{'Mark Incomplete' if task['completed'] else 'Mark Complete'} {i}"):
                            task['completed'] = not task['completed']
                            save_data()
                            st.rerun()
                    with col_b:
                        if st.button(f"Delete Task {i}"):
                            st.session_state.plans[username].remove(task)
                            save_data()
                            st.rerun()
    else:
        st.info("No tasks yet. Start planning today!")

def resources_page():
    st.header("📚 Mental Health Resources")
    
    # Books section
    st.subheader("📖 Recommended Books")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTVUs_oFTTue2AB5HqZ1kbgfYlAXm0AyPtB_Q&s", use_container_width=True)
        st.write("**[The Anxiety and Phobia Workbook](https://www.amazon.com/Anxiety-Phobia-Workbook-Edmund-Bourne/dp/1626252157/)**")
        st.write("By Edmund J. Bourne")
        st.write("A comprehensive guide for managing anxiety and phobias.")
    
    with col2:
        st.image("https://img.freepik.com/free-vector/world-mental-health-day_23-2148636755.jpg", use_container_width=True)
        st.write("**[Feeling Good: The New Mood Therapy](https://www.amazon.com/Feeling-Good-New-Mood-Therapy/dp/0380810336/)**")
        st.write("By David D. Burns")
        st.write("Classic self-help book based on cognitive behavioral therapy.")
    
    with col3:
        st.image("https://fourminutes.training/wp-content/uploads/2020/10/mental-health-matters-scaled.jpg", use_container_width=True)
        st.write("**[The Mindfulness and Acceptance Workbook](https://www.amazon.com/Mindfulness-Acceptance-Workbook-Anxiety-Commitment/dp/1626253358/)**")
        st.write("By John P. Forsyth and Georg H. Eifert")
        st.write("Practical exercises for emotional well-being.")
    
    # Articles section
    st.subheader("📝 Articles to Read")

    articles = [
        {
            "title": "The Science of Happiness",
            "link": "https://greatergood.berkeley.edu/podcasts/series/the_science_of_happiness",
            "image": "https://img.onmanorama.com/content/dam/mm/en/lifestyle/health/images/2023/3/4/mental-health-new-c.jpg?crop=fc&w=575&h=575",
            "description": "Explore research-backed techniques to improve happiness and well-being."
        },
        {
            "title": "How to Reduce Anxiety Naturally",
            "link": "https://www.verywellmind.com/tips-to-reduce-anxiety-5176066",
            "image": "https://media.istockphoto.com/id/1301653190/vector/anxiety-anxious-teen-girl-suffering-from-depression-sitting-with-head-in-lap.jpg?s=612x612&w=0&k=20&c=WWtJES8xpaF2PsbbvcSWYF0z4yIDd1TkuEFyCoWNFPU=",
            "description": "Learn natural and effective ways to manage anxiety through lifestyle changes."
        },
        {
            "title": "Benefits of Meditation for Mental Health",
            "link": "https://www.verywellmind.com/the-benefits-of-meditation-5201607",
            "image": "https://i.insider.com/615f0c74c2a4ca0018760478?width=700",
            "description": "Understand how meditation can reduce stress, enhance focus, and boost emotional health."
        },
        {
            "title": "How to Build Healthy Habits",
            "link": "https://jamesclear.com/habits",
            "image": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRii-OrolXcvf5OpROGMpJlwJuUrUAew5i3BQ&s",
            "description": "Discover the science behind habit formation and how to build lasting positive habits."
        },
        {
            "title": "Cognitive Behavioral Therapy Explained",
            "link": "https://www.verywellmind.com/what-is-cognitive-behavior-therapy-2795747",
            "image": "https://img.freepik.com/free-vector/flat-people-with-mental-health-problems_23-2149059511.jpg",
            "description": "A deep dive into CBT, its principles, and how it helps in overcoming negative thought patterns."
        }
    ]
    
    # Display articles in a structured container layout
    for article in articles:
        with st.container():
            col1, col2 = st.columns([1, 2])  # Image on the left, text on the right
            with col1:
                st.image(article["image"], use_container_width=True)
            with col2:
                st.markdown(f"### [{article['title']}]({article['link']})")
                st.write(article["description"])  # Add brief description here
                st.write("Click the title to read the full article.")
            st.markdown("---")

    # Exercises section
    st.subheader("🏋️ Helpful Exercises")
    
    with st.expander("🫁 Deep Breathing Exercise"):
        st.write("""
        1. Find a comfortable position.
        2. Breathe in slowly through your nose for 4 counts.
        3. Hold your breath for 2 counts.
        4. Exhale slowly through your mouth for 6 counts.
        5. Repeat for 5 minutes.
        """)
    
    with st.expander("🌎 5-4-3-2-1 Grounding Technique"):
        st.write("""
        When feeling anxious, identify:
        - 5 things you can see
        - 4 things you can touch
        - 3 things you can hear
        - 2 things you can smell
        - 1 thing you can taste
        """)

    with st.expander("💪 Progressive Muscle Relaxation"):
        st.write("""
        1. Tense the muscles in your toes for 5 seconds.
        2. Release and notice the difference.
        3. Move up to your calves, then thighs, and so on.
        4. Work your way up through your entire body.
        """)
    
    # Crisis resources
    st.subheader("🚨 Crisis Resources")
    st.info("""
    **Remember**: If you're in crisis, please reach out for professional help:
    
    - **[National Suicide Prevention Lifeline](https://988lifeline.org/)**: 988 or 1-800-273-8255  
    - **[Crisis Text Line](https://www.crisistextline.org/)**: Text HOME to 741741  
    - **[Find a Therapist](https://www.psychologytoday.com/us/therapists)**: Psychology Today Directory  

    This app is not a substitute for professional mental health care.
    """)


def meditation_page():
    st.header("Meditation Corner")
    
    username = st.session_state.current_user
    
    # Meditation timer
    st.subheader("Meditation Timer")
    
    meditation_time = st.radio("Choose meditation duration:", [1, 3, 5, 10], horizontal=True)
    
    start_col, stop_col = st.columns(2)
    
    if 'meditation_running' not in st.session_state:
        st.session_state.meditation_running = False
    if 'meditation_start_time' not in st.session_state:
        st.session_state.meditation_start_time = None
    if 'meditation_duration' not in st.session_state:
        st.session_state.meditation_duration = 0
    
    with start_col:
        if st.button("Start Meditation"):
            st.session_state.meditation_running = True
            st.session_state.meditation_start_time = time.time()
            st.session_state.meditation_duration = meditation_time * 60
    
    with stop_col:
        if st.button("Stop Meditation"):
            if st.session_state.meditation_running:
                elapsed_time = time.time() - st.session_state.meditation_start_time
                
                # Record meditation session
                if username not in st.session_state.meditation_logs:
                    st.session_state.meditation_logs[username] = []
                
                st.session_state.meditation_logs[username].append({
                    'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'duration_seconds': int(elapsed_time),
                    'planned_minutes': meditation_time
                })
                save_data()
                
                st.session_state.meditation_running = False
                st.success(f"Meditation completed! You meditated for {int(elapsed_time)} seconds.")
    
    # Display timer if meditation is running
    if st.session_state.meditation_running:
        elapsed = time.time() - st.session_state.meditation_start_time
        remaining = max(0, st.session_state.meditation_duration - elapsed)
        
        progress_pct = min(100, (elapsed / st.session_state.meditation_duration) * 100)
        st.progress(progress_pct / 100)
        
        st.metric("Remaining Time", f"{int(remaining // 60)}:{int(remaining % 60):02d}")
        
        # Auto-stop when timer is complete
        if remaining <= 0 and st.session_state.meditation_running:
            st.balloons()
            st.success(f"Meditation complete! Well done!")
            
            # Record meditation session
            if username not in st.session_state.meditation_logs:
                st.session_state.meditation_logs[username] = []
            
            st.session_state.meditation_logs[username].append({
                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'duration_seconds': st.session_state.meditation_duration,
                'planned_minutes': meditation_time
            })
            save_data()
            
            st.session_state.meditation_running = False
    
    # Meditation history
    st.subheader("My Meditation History")
    
    if username in st.session_state.meditation_logs and st.session_state.meditation_logs[username]:
        meditation_data = st.session_state.meditation_logs[username]
        meditation_data.sort(key=lambda x: x['date'], reverse=True)
        
        # Calculate total meditation time
        total_seconds = sum(session['duration_seconds'] for session in meditation_data)
        total_minutes = total_seconds / 60
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Sessions", len(meditation_data))
        with col2:
            st.metric("Total Meditation Time", f"{int(total_minutes)} minutes")
        
        # Create chart data
        dates = []
        durations = []
        last_7_sessions = meditation_data[:7]
        
        for session in reversed(last_7_sessions):
            dates.append(session['date'])
            durations.append(session['duration_seconds'] / 60)
        
        # Display history in a table
        history_df = pd.DataFrame(meditation_data)
        history_df['duration_minutes'] = history_df['duration_seconds'] / 60
        history_df = history_df[['date', 'planned_minutes', 'duration_minutes']]
        history_df.columns = ['Date', 'Planned (min)', 'Actual (min)']
        history_df['Actual (min)'] = history_df['Actual (min)'].round(1)
        
        st.write("Recent meditation sessions:")
        st.dataframe(history_df.head(10), use_container_width=True)
    else:
        st.info("No meditation sessions recorded yet. Start your first session today!")

def support_forum():
    st.header("💬 Anonymous Support Forum")
    
    # Initialize forum data if not present
    if "forum_posts" not in st.session_state:
        st.session_state.forum_posts = []

    # Text area for posting
    post_text = st.text_area("Share your thoughts or ask a question (Anonymous):", height=100)

    if st.button("Post"):
        if post_text.strip():
            new_post = {
                "id": len(st.session_state.forum_posts) + 1,
                "text": post_text,
                "replies": []
            }
            st.session_state.forum_posts.insert(0, new_post)  # Insert at the top
            st.success("Your post has been added!")
            st.rerun()

    st.subheader("📌 Community Posts")
    
    if not st.session_state.forum_posts:
        st.info("No posts yet. Start the conversation!")
    else:
        for post in st.session_state.forum_posts:
            with st.expander(f"📝 Post #{post['id']}"):
                st.write(post["text"])
                
                # Reply section
                reply_text = st.text_area(f"Reply to Post #{post['id']}", key=f"reply_{post['id']}")
                if st.button(f"Reply to Post {post['id']}"):
                    if reply_text.strip():
                        post["replies"].append(reply_text)
                        st.success("Reply added!")
                        st.rerun()
                
                # Show replies
                if post["replies"]:
                    st.subheader("Replies:")
                    for reply in post["replies"]:
                        st.write(f"💬 {reply}")
                        # Main ap

def main():
    load_data()
    
    # Define sidebar
    st.sidebar.title("🧘MindCare\n  Your Mental Health Companion")
    
    # Profile section in sidebar (if logged in)
    if st.session_state.authenticated:
        username = st.session_state.current_user
        st.sidebar.markdown("---")
        st.sidebar.subheader("👤 My Profile")
        st.sidebar.write(f"**Name**: {username}")
        st.sidebar.write(f"**Email**: {st.session_state.users[username]['email']}")
        st.sidebar.write(f"🔥 **Streak**: {st.session_state.check_in_streak.get(username, 0)} days")

        # Daily Check-in & Affirmation
        st.sidebar.markdown("---")
        st.sidebar.subheader("📅 Daily Check-in")
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")

        if "last_check_in_date" not in st.session_state:
            st.session_state.last_check_in_date = ""

        if st.session_state.last_check_in_date != today_date:
            if st.button("✅ Check-in for Today"):
                st.session_state.last_check_in_date = today_date
                st.session_state.check_in_streak[username] = st.session_state.check_in_streak.get(username, 0) + 1
                save_data()
                st.success("Check-in complete! Keep up the great work!")

        st.sidebar.write("💡 **Today's Affirmation:**")
        affirmations = [
            "You are doing your best, and that is enough.",
            "Your feelings are valid.",
            "Small progress is still progress.",
            "You deserve peace and happiness.",
            "Breathe. You got this."
        ]
        st.sidebar.info(random.choice(affirmations))
    # Navigation
    if st.session_state.authenticated:
        page = st.sidebar.radio("📍 Navigation", [
            "AI Assistant", "Journal", "Planner", "Resources", "Meditation", "Support Forum"
        ])

        st.sidebar.subheader("🎯 Today's Self-Care Challenge")
        st.sidebar.info(generate_daily_challenge())
        st.sidebar.subheader("🎵 Relaxing Music for You")
        mood_summary, _ = analyze_mood(st.session_state.current_user)
        mood = mood_summary.split("**")[1] if "**" in mood_summary else "Neutral"
        music_file = get_music_recommendation(mood)

        if music_file and os.path.exists(music_file):
            st.sidebar.audio(music_file, format="audio/mp3")
        else:
            st.sidebar.info("No music available for this mood. Try uploading your own.")
        
        if st.sidebar.button("🚪 Logout"):
            logout()
        
        # Display selected page
        if page == "AI Assistant":
            ai_assistant_page()
        elif page == "Journal":
            journal_page()
        elif page == "Planner":
            planner_page()
        elif page == "Resources":
            resources_page()
        elif page == "Meditation":
            meditation_page()
        elif page == "Support Forum":
            support_forum()
    else:
        auth_option = st.sidebar.radio("", ["Login", "Register"])
        
        if auth_option == "Login":
            login()
        else:
            register()
        
        # Welcome screen for non-authenticated users
        st.title("🌿 Welcome to MindCare")
        st.write("Your personal mental health companion")

        st.info("Please login or register to access all features.")
        
        st.markdown("""
        ## ✨ Features:
        - 🤖 **AI Assistant**: Get instant mental health support.
        - 📔 **Personal Journal**: Track your thoughts & emotions.
        - 📅 **Planner & Task Manager**: Organize your day with AI-powered prioritization.
        - 📚 **Mental Health Resources**: Curated books, exercises, & crisis support.
        - 🧘 **Meditation Corner**: Timer & history to build a mindfulness habit.
        - 💬 **Anonymous Support Forum**: Share your thoughts & get peer support.
        - ✅ **Daily Check-in & Affirmations**: Build healthy habits & stay motivated.
        - 📊 **Mood Analysis & Self-Care Suggestions**: AI-powered mood tracking for personalized self-care.
        
        Your data is stored locally and remains private. 💙
        """)

if __name__ == "__main__":
    main()