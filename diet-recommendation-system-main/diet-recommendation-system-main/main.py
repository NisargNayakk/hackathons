from flask import Flask, render_template, request
import pickle
import pandas as pd
import csv
import os

app = Flask(__name__)

# Check if the model file exists, if not, create a dummy model for demonstration
model_path = 'food_model.pickle'
if not os.path.exists(model_path):
    print("Warning: food_model.pickle not found. Creating a dummy model.")
    from sklearn.ensemble import RandomForestClassifier
    dummy_model = RandomForestClassifier()
    dummy_model.fit([[0, 0, 0]], ['Weight_Loss'])  # Dummy training
    with open(model_path, 'wb') as file:
        pickle.dump(dummy_model, file)

# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Check if the CSV file exists, if not, create a dummy DataFrame
csv_path = 'done_food_data.csv'
if not os.path.exists(csv_path):
    print("Warning: done_food_data.csv not found. Creating a dummy DataFrame.")
    food_data = pd.DataFrame({
        'Descrip': ['Apple', 'Banana', 'Chicken', 'Spinach', 'Egg'],
        'category': ['Weight_Loss', 'Weight_Gain', 'Muscle_Gain', 'Weight_Loss', 'Muscle_Gain'],
        'Iron_mg': [0.1, 0.3, 1.0, 2.7, 1.8],
        'Calcium_mg': [5, 5, 11, 99, 56]
    })
    food_data.to_csv(csv_path, index=False)
else:
    food_data = pd.read_csv(csv_path)

def read_csv(file_path, sort_by='Descrip'):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        sorted_rows = sorted(rows, key=lambda x: x[sort_by])
        return sorted_rows

@app.route("/")
def index():
    return render_template("mainpage.html")

@app.route("/predict", methods=['POST'])
def predict():
    input_1 = float(request.form['input_1'])
    input_2 = float(request.form['input_2'])
    input_3 = float(request.form['input_3'])
    
    inputs = [[input_1, input_2, input_3]]
    
    prediction = model.predict(inputs)
    
    if prediction[0] == 'Muscle_Gain':
        result = 'Muscle Gain'
    elif prediction[0] == 'Weight_Gain':
        result = 'Weight Gain'
    elif prediction[0] == 'Weight_Loss':
        result = 'Weight Loss'
    else:
        result = 'General food'
    
    return render_template("mainpage.html", result=result)

@app.route("/musclegain", methods=['POST'])
def musclegain():
    vegetarian = request.form.getlist('vegetarian')
    iron = request.form.getlist('iron')
    calcium = request.form.getlist('calcium')
    anyfoods = request.form.getlist('anyfoods')

    muscle_gain_data = food_data[food_data['category'] == 'Muscle_Gain']

    if 'iron' in iron:
        muscle_gain_data = muscle_gain_data[muscle_gain_data['Iron_mg'] > 6]
    if 'calcium' in calcium:
        muscle_gain_data = muscle_gain_data[muscle_gain_data['Calcium_mg'] > 150]
    if 'vegetarian' in vegetarian:
        exclude_keywords = ['Egg','Fish', 'meat', 'beef','Chicken','Beef','Deer','lamb','crab','pork','Frog legs','Pork','Turkey','flesh','Ostrich','Emu','cuttelfish','Seaweed','crayfish','shrimp','Octopus']
        muscle_gain_data = muscle_gain_data[~muscle_gain_data['Descrip'].str.contains('|'.join(exclude_keywords), case=False)]

    musclegainfoods = muscle_gain_data['Descrip'].sample(n=min(5, len(muscle_gain_data))).to_string(index=False)
    
    return render_template("mainpage.html", musclegainfoods=musclegainfoods)

@app.route("/weightgain", methods=['POST'])
def weightgain():
    vegetarian = request.form.getlist('vegetarian')
    iron = request.form.getlist('iron')
    calcium = request.form.getlist('calcium')
    anyfoods = request.form.getlist('anyfoods')

    weight_gain_data = food_data[food_data['category'] == 'Weight_Gain']

    if 'iron' in iron:
        weight_gain_data = weight_gain_data[weight_gain_data['Iron_mg'] > 6]
    if 'calcium' in calcium:
        weight_gain_data = weight_gain_data[weight_gain_data['Calcium_mg'] > 150]
    if 'vegetarian' in vegetarian:
        exclude_keywords = ['Egg','Fish', 'meat', 'beef','Chicken','Beef','Deer','lamb','crab','pork','turkey','flesh']
        weight_gain_data = weight_gain_data[~weight_gain_data['Descrip'].str.contains('|'.join(exclude_keywords), case=False)]

    weightgainfoods = weight_gain_data['Descrip'].sample(n=min(5, len(weight_gain_data))).to_string(index=False)
    
    return render_template("mainpage.html", weightgainfoods=weightgainfoods)

@app.route("/weightloss", methods=['POST'])
def weightloss():
    vegetarian = request.form.getlist('vegetarian')
    iron = request.form.getlist('iron')
    calcium = request.form.getlist('calcium')
    anyfoods = request.form.getlist('anyfoods')

    weight_loss_data = food_data[food_data['category'] == 'Weight_Loss']

    if 'iron' in iron:
        weight_loss_data = weight_loss_data[weight_loss_data['Iron_mg'] > 6]
    if 'calcium' in calcium:
        weight_loss_data = weight_loss_data[weight_loss_data['Calcium_mg'] > 150]
    if 'vegetarian' in vegetarian:
        exclude_keywords = ['Egg','Fish', 'meat', 'beef','Chicken','Beef','Deer','lamb','crab','pork','turkey','flesh']
        weight_loss_data = weight_loss_data[~weight_loss_data['Descrip'].str.contains('|'.join(exclude_keywords), case=False)]

    weightlossfoods = weight_loss_data['Descrip'].sample(n=min(5, len(weight_loss_data))).to_string(index=False)
    
    return render_template("mainpage.html", weightlossfoods=weightlossfoods)

@app.route("/search", methods=['POST'])
def search():
    sort_by = request.form.get('sort_by', 'Descrip')
    rows = read_csv('done_food_data.csv', sort_by)
    return render_template('search.html', rows=rows)

if __name__ == "__main__":
    app.run(debug=True)