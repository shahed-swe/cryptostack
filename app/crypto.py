from flask import (
    Blueprint,
    render_template,
    request,
)
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
today = date.today()
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
import warnings
from numpy import array
warnings.filterwarnings('ignore')
from keras.callbacks import ModelCheckpoint  

crypto = Blueprint('crypto', __name__)

@crypto.route("/analyse")
def analyse():
    return render_template("analyse.html")

@crypto.route('/test_get_plot', methods=['POST','GET'])
def get_plot():
    # Same cool stuff here.
    if request.method == 'POST':
        global a, c
        a = request.form.get('crypto')
        c = "-usd"
        ac = a.strip('"') + c
        d1 = today.strftime("%Y-%m-%d")
        end_date = d1
        d2 = date.today() - timedelta(days=730)
        d2 = d2.strftime("%Y-%m-%d")
        start_date = d2

        data1 = yf.download(str(ac), 
                            start=start_date, 
                            end=end_date, 
                            progress=False)
        data1["Date"] = data1.index
        data1 = data1[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        data1.reset_index(drop=True, inplace=True)
        data1 = pd.DataFrame(data1)
        data = data1.Open
        data = data.apply(np.log)
        data = data.apply(np.sqrt)
            
        n_steps = 20
        # split into samples
        X, y = split_sequence(data, n_steps)
            
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
       
        checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,
            save_best_only=True, mode='auto', period=1)
        model = Sequential()
        tf.keras.backend.clear_session()
        model=Sequential()
        model.add((LSTM(32,return_sequences=True, input_shape=(n_steps, n_features),)))
        model.add((LSTM(16)))
        model.add((Dense(5)))
        model.add((Dense(1)))

        model.compile(loss='mean_squared_error',optimizer='adam')

        model.fit(X, y, 
            validation_data=(X, y),
            epochs=3, verbose=2, batch_size = 12,callbacks=[checkpoint])
        
        from keras.models import load_model
        model=load_model('best_model.hdf5')

        n_steps=20
        v = data[len(data)-21:len(data)]
        i=0
        predicted=model.predict(X)
      
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(list(predicted[600:]), linestyle='dashed', color='red')
        ax.plot(list(y[600:]), color='blue')
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.grid()
        ax.legend(('Predicted Trend','Actual Trend'))
        plt.savefig('app/static/media/predict.png')

        i=0
        global lst_output
        lst_output=[]
        x_input = np.array(v)
        temp_input=list(x_input)
        while(i<15):
            
            if(len(temp_input)>3):
                x_input=array(temp_input[1:])
                x_input=pd.DataFrame(x_input)
                x_input=np.array(x_input)
                x_input = x_input.reshape((1, n_steps, n_features))
                x_input=np.asarray(x_input).astype(np.float32)
                yhat = model.predict(x_input, verbose=1)
                temp_input.append(yhat[0][0])
                temp_input=temp_input[1:]
                lst_output.append(yhat[0][0])
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
            
                temp_input.append(yhat[0][0])
                lst_output.append(yhat[0][0])
                i=i+1
            
        global b
        data=data[500:]
        b = list(data)+list(lst_output)
        img = io.BytesIO()
        fig,ax = plt.subplots(figsize=(10,5))
        ax.plot(list(b),linestyle='dashed', color='red')
        ax.plot(list(data),color='blue')
        ax.set_xlabel("Time (Days)")
        ax.set_ylabel("Value")
        ax.grid()
        ax.legend(('Predicted Trend','Actual Trend'))
        plt.savefig('app/static/media/my_plot.png')
        img.seek(0)
 
        gv = pd.date_range(str(d1), periods=15)
        lst_output = pd.DataFrame(lst_output)
        pred_log = lst_output.apply(np.square)
        pred = pred_log.apply(np.exp)
        dv = pred[0]
        gv = pd.DataFrame(gv)
        gv['Forecast'] = dv
        gv = gv.rename(columns={gv.columns[0]: 'Days', 'Forecast': ''})
        gv.set_index('Days', inplace=True)
        gv.to_csv('app/forecast.csv')
            
    return render_template('analyse.html', get_plot = True, plot_url = 'static/media/my_plot.png', plot_url1 = 'static/media/predict.png', data = data,tables=[gv.to_html(classes='data')], titles=gv.columns.values)

from flask import Flask,render_template,send_file
@crypto.route('/download_file')
def download_file():
    path = "forecast.csv"
    return send_file(path,as_attachment=True)

# initialise scraper without time interval for max historical data

from numpy import array
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)