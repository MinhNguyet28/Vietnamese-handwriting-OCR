from flask import Flask, Blueprint, jsonify, render_template, request,make_response
# from model import ModelPredict
app = Flask(__name__)

# modelPredict = ModelPredict()
@app.route('/')
def index():
   return render_template('draw_page.html') 

@app.route('/draw_page')
def draw_page():
    return render_template('draw_page.html')

@app.route('/upload', methods = ['POST'])
def upload():
   data = request.json
   # predict = modelPredict.predict(data['0']['image'])
   # return jsonify(index=0,predict=predict)
   return jsonify(index=0,predict='Predicted Text')


if __name__ == '__main__':
    app.run(host= '127.0.0.1', port=5000, debug=False)
   #  app.run(host= '192.168.1.15', port=5000, debug=False)