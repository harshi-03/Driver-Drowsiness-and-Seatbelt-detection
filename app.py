from flask import Flask, render_template, request
import subprocess

app = Flask(__name__)

# Define a route to render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle script execution
@app.route('/run_script', methods=['POST'])
def run_script():
    if request.method == 'POST':
        # Run your Python script
        subprocess.Popen(['python', 'FinalRun.py'])
        return "Script is running..."

if __name__ == '__main__':
    app.run(debug=True)
