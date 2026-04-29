import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
import warnings
import json

warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Data Initialization ---
try:
    # Loading the dataset
    zomato_df = pd.read_csv("restaurant1.csv")
    df_percent = zomato_df.copy()
    
    # Setting index for the recommendation engine
    df_percent.set_index('name', inplace=True)
    indices = pd.Series(df_percent.index)

    # Loading the pre-computed TF-IDF Cosine Similarity Matrix
    with open("restaurant.pkl", "rb") as f:
        cosine_similarities = pickle.load(f)

except FileNotFoundError:
    print("⚠️ Error: Critical data files (CSV or PKL) missing. Please check your directory.")
except Exception as e:
    print(f"⚠️ An unexpected error occurred during startup: {e}")

# --- Core Inference Logic ---
def recommend(name, cosine_similarities=cosine_similarities):
    recommend_restaurant = []
    
    # Boundary Check: If the restaurant isn't in our vector space
    if name not in indices.values:
        return pd.DataFrame(columns=['Error'])

    # Find the index of the user's input restaurant
    idx = indices[indices == name].index[0]
    
    # Extract similarity scores and sort them
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 similar nodes (skipping the first one as it's the input itself)
    top_indexes = list(score_series.iloc[1:min(31, len(score_series))].index)

    # Map indexes back to restaurant names
    for each in top_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])

    # Build the recommendation DataFrame
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    for each in recommend_restaurant:
        selected_data = df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each]
        if not selected_data.empty:
            # We take a sample of 1 in case of duplicate names in the original CSV
            df_new = pd.concat([df_new, selected_data.sample(n=1)])

    # Clean up results
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)

    # --- Aura Gastronomica Professional Mapping ---
    df_new.rename(columns={
        'cuisines': 'FLAVOR_PROFILE',
        'Mean Rating': 'CONFIDENCE_SCORE',
        'cost': 'ESTIMATED_VALUE'
    }, inplace=True)

    return df_new

# --- Routes ---

@app.route('/')
def home():
    """Landing page for Aura Gastronomica."""
    return render_template('index.html')

@app.route('/web')
def web():
    """The Neural Inference Engine search interface."""
    return render_template('web.html')

@app.route('/result', methods=['POST'])
def result():
    """Processes the recommendation request and renders the dashboard."""
    output = request.form.get('output', '')
    
    if not output:
        return render_template('web.html') # Redirect back if search is empty

    res = recommend(output)

    # Convert the resulting DataFrame to a sleek HTML table
    html_table = res.to_html(
        classes='custom-table',
        justify='left',
        border=0,
        index=True,
        index_names=False
    )

    # Prepare JSON data for the dynamic DNA tags and UI elements
    if not res.empty:
        names = res.index.tolist()
        ratings = res['CONFIDENCE_SCORE'].tolist()
        cost = res['ESTIMATED_VALUE'].tolist()
    else:
        names, ratings, cost = [], [], []

    return render_template(
        'result.html',
        keyword=html_table,
        search_term=output,
        names=json.dumps(names),
        ratings=json.dumps(ratings),
        cost=json.dumps(cost)
    )

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """Provides real-time search suggestions as the user types."""
    search_term = request.args.get('term', '').lower()
    if not search_term:
        return jsonify([])

    # Filter for matches within the restaurant index
    matches = df_percent[
        df_percent.index.str.lower().str.contains(search_term, na=False)
    ].index.tolist()

    # Remove duplicates and limit to top 10 for performance
    matches = sorted(list(set(matches)))
    return jsonify(matches[:10])

# --- Entry Point ---
if __name__ == '__main__':
    # Running on debug mode for development; turn off in production.
    app.run(debug=True)