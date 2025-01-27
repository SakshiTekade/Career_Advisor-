from flask import Flask, request, render_template
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv(r"C:\Users\saksh\OneDrive\Desktop\Dataset\career_recommendation_data.csv")

# Preprocess data
data['combined'] = data['interest'] + " " + data['skills']

# Initialize vectorizer
vectorizer = TfidfVectorizer()

# Train vectorizer on dataset
tfidf_matrix = vectorizer.fit_transform(data['combined'])

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
        <title>AI-Based Career Advisor</title>
        <style>
            body {
                background: linear-gradient(to right, #667eea, #764ba2);
                color: white;
                font-family: 'Roboto', sans-serif;
            }
            .container {
                margin-top: 5%;
                background: white;
                color: #333;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
            }
            h1 {
                color: #5a67d8;
                font-weight: bold;
            }
            .btn-primary {
                background-color: #5a67d8;
                border-color: #5a67d8;
                transition: all 0.3s ease;
            }
            .btn-primary:hover {
                background-color: #434190;
                border-color: #434190;
            }
            .form-control {
                border-radius: 8px;
            }
            footer {
                margin-top: 50px;
                text-align: center;
                font-size: 14px;
                color: white;
            }
            footer a {
                color: #f4f4f9;
                text-decoration: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="text-center mb-4"><i class="fas fa-lightbulb"></i> AI-Based Career Advisor</h1>
            <form action="/recommend" method="post">
                <div class="mb-3">
                    <label for="interest" class="form-label">Your Interests:</label>
                    <input type="text" id="interest" name="interest" class="form-control" placeholder="e.g., web development">
                </div>
                <div class="mb-3">
                    <label for="skills" class="form-label">Your Skills:</label>
                    <input type="text" id="skills" name="skills" class="form-control" placeholder="e.g., HTML, CSS, JavaScript">
                </div>
                <button type="submit" class="btn btn-primary btn-block w-100">Get Career Advice</button>
            </form>
        </div>
        <footer>
            <p>Developed with <i class="fas fa-heart"></i> Sakshi | <a href="#">GitHub</a></p>
        </footer>
    </body>
    </html>
    '''

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get user input
        user_interest = request.form['interest']
        user_skills = request.form['skills']
        
        # Handle case where user doesn't provide input
        if not user_interest or not user_skills:
            return '''
            <div class="container">
                <h1 class="text-center text-danger">Error</h1>
                <p class="text-center">Both interest and skills are required. Please go back and try again.</p>
                <div class="text-center">
                    <a href="/" class="btn btn-primary mt-4">Go Back</a>
                </div>
            </div>
            '''
        
        # Combine user input
        user_input = user_interest + " " + user_skills
        
        # Transform user input using vectorizer
        user_tfidf = vectorizer.transform([user_input])
        
        # Calculate similarity
        similarity = cosine_similarity(user_tfidf, tfidf_matrix)
        
        # Find the best match
        best_match_idx = similarity.argmax()
        recommended_career = data.iloc[best_match_idx]['career']
        
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
            <title>Career Recommendation</title>
            <style>
                body {{
                    background: linear-gradient(to right, #667eea, #764ba2);
                    color: white;
                    font-family: 'Roboto', sans-serif;
                }}
                .container {{
                    margin-top: 5%;
                    background: white;
                    color: #333;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0px 10px 20px rgba(0, 0, 0, 0.2);
                }}
                h1 {{
                    color: #5a67d8;
                    font-weight: bold;
                }}
                .btn-primary {{
                    background-color: #5a67d8;
                    border-color: #5a67d8;
                    transition: all 0.3s ease;
                }}
                .btn-primary:hover {{
                    background-color: #434190;
                    border-color: #434190;
                }}
                footer {{
                    margin-top: 50px;
                    text-align: center;
                    font-size: 14px;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center">Career Recommendation</h1>
                <p class="text-center">Based on your interests and skills, we recommend:</p>
                <h2 class="text-center text-primary">{recommended_career}</h2>
                <div class="text-center">
                    <a href="/" class="btn btn-primary mt-4">Try Again</a>
                </div>
            </div>
            <footer>
                <p>Developed with <i class="fas fa-heart"></i> Sakshi | <a href="#">GitHub</a></p>
            </footer>
        </body>
        </html>
        '''
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
