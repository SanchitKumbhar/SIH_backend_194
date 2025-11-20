from flask import Flask
import sys

print("--- DEBUG: Starting the Flask App Execution ---", file=sys.stderr)
app= Flask (__name__)
@app.route('/')
def hello():
    return "<h1>Hello, World!</h1>"

@app.route('/about')
def about_page():
    return "<h1>About Page</h1>"

@app.route('/about/<username>')
def about_user(username):
    return f'<h1>this is about page of {username}</h1>'

@app.route('/loop/<int:num>')
def loop_example(num):
    result = ''
    for i in range (num):
        result +=f'<p>Iteration {i+1}</p>'
    return result
    

if __name__ == '__main__':
    print("--- DEBUG: Entering app.run() ---", file=sys.stderr)
    app.run(debug=True)