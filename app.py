import streamlit as st
import base64

st.title('ML- Mini Project')

st.header('Stock Market Prediction')

with st.expander('Submitted by: '):
    st.write('Advait Shinde    - 41058')
    st.write('Sakshi Ingale    - 41021')
    st.write('Gayatri Kurulkar - 41039')
    st.write('Yash Jangale     - 41027')

st.subheader('Problem Statement: ')
st.write ('Use the following dataset to analyze ups and downs in the market and predict future stock price returns based on Indian Market data from 2000 to 2020.')

st.subheader('Dataset Details: ')
st.write('''Data Structure: 
         
**Index:**

The index is a series of dates, which represents the trading days. 
It ranges from August 28, 2001 to June 5, 2020 in the snippet provided, and each entry corresponds to a specific trading day. 
         
**Columns:** There are six columns in the dataset, all of which contain numerical values:

**Open:** The opening price of the stock on a given day (e.g., $7.19 on January 3, 2000).
         
**High:** The highest price reached by the stock during the trading day (e.g., $7.27 on January 3, 2000).
         
**Low:** The lowest price reached by the stock during the trading day (e.g., $6.90 on January 3, 2000).
         
**Close:** The closing price of the stock on the trading day (e.g., $7.20 on January 3, 2000).
         
**Adj Close:** The adjusted closing price, which accounts for corporate actions like stock splits and dividends. This is typically used for more accurate analysis of price trends over time.

**Volume:** The number of shares traded on that day (e.g., 1,080,680 on January 3, 2000).''')

st.subheader('Implementation')

with st.expander('Dataset Exploration :'):
    st.write('Dataframe Head: ')
    st.image('plots_and_ss/image1.png')

    st.write('Dataframe Shape: ')
    st.write('(3993, 6)')

    st.write('Dataset Description : ')
    st.image('plots_and_ss/image2.png')

    st.write('Dataframe Information: ')
    st.image('plots_and_ss/image3.png')

    st.write('Target Variable (Closing Price) vs Adj Close of Random Company from BSE')
    st.image('plots_and_ss/output1.png')

    st.write('Number of stocks sold by a Random Company ')
    st.image('plots_and_ss/output2.png')

with st.expander('Feature Engineering: '):
    st.subheader('1. Moving Average (MA) Feature Addition')
    st.markdown('''These values represent moving average (MA) periods in terms of trading days. The moving average is commonly used in financial analysis to smooth out price data and identify trends over different time frames. 
                
    10-day MA: A short-term trend indicator (often used for immediate trading decisions).
                
    50-day MA: A medium-term trend.
    
    100-day and 365-day MAs: Longer-term trends, with 365 days typically representing a full year's worth of trading data.''')

    st.write ('**Sample plot of Random Stock MA for 10 days**')
    st.image('plots_and_ss/output3.png')

    st.write('**Sample plot of Random Stock MA for 365 days** ')
    st.image('plots_and_ss/output5.png')

    st.write('All MA combined: ')
    st.image('plots_and_ss/output6.png')

    st.write(' ')

    st.subheader('2. Daily Return Feature : ')
    st.markdown('''Daily Return= 

(Current Day Adj Close - Previou Days Adj Close) / Previous Day's Adj Close x 100''')
    
with st.expander('Building the Model : RNN '):
    st.write('Using the LSTM - Long Short-Term Memory from Keras Models')

    st.subheader('''Step 1: Import Libraries:

    from keras.models import Sequential
                    
    from keras.layers import Dense, LSTM
                    
    import tensorflow as tf
                    ''')
    
    st.subheader('Step 2: Prepare Training DataSet')

    st.write('Takes in a DataFrame and selects only the Close price column to forecast future prices.')

    st.write('''Code:
             
    def build_training_dataset(input_ds):
             
    # new df with close column
             
    input_ds.reset_index()
    data = input_ds.filter(items=['Close'])
    
    dataset = data.values

    # number of rows to train the model on
    
    training_data_len = int(np.ceil( len(dataset) * .95 ))
    return data, dataset, training_data_len

''')

    st.subheader('Step3 : Pre-processing the Data - Scaling and Splitting.')

    st.write('''Code for Scaling:
             
    from sklearn.preprocessing import MinMaxScaler
    def scale_the_data(dataset):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        return scaler, scaled_data
''')

    st.write('Used Min Max Scaler to scale variables betwen the range 0 and 1')

    st.write('''Code for Splitting:
             
    def split_train_dataset(training_data_len):
    train_data = scaled_data[0:int(training_data_len), :]
             
    # split the data into x_train and y_train data sets
             
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            #print(x_train)
            #print(y_train)
            print('.')
             
''')

    st.write('''Splits the scaled training data into features (x_train) and labels (y_train).
    For each time step, the function uses the previous 60 days of data to predict the next day's value (sliding window technique).''')

    st.subheader('Step 4: Building the LSTM Model ')

    st.write('''Code:
             
    def build_lstm_model(x_train,y_train):

    # Build the LSTM model

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # compile the model
    # adam ~ Stochastic Gradient descent method.
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model 
''')
    
    st.write('''
             Builds a sequential LSTM model using Keras.

The first LSTM layer has 128 units with return_sequences=True because another LSTM layer follows it.
             
The second LSTM layer has 64 units, followed by two dense layers for regression output.
             
The model is compiled using the Adam optimizer and the mean squared error loss function.''')

    st.subheader('Step 6: Predictions!')
    st.write('Create Test Dataset and Predict the Results.')


with st.expander('Results:'):
    c1, c2 = st.columns(2)

    with st.container():
        c1.subheader('Close Price Predictions')
        c1.image('results/output1.png')
        c1.write(' RMSE: 1.215 ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.image('results/output2.png')
        c1.write(' RMSE: 3.660 ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.image('results/output3.png')
        c1.write(' RMSE: 0.194 ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.image('results/output4.png')
        c1.write(' RMSE: 0.181 ')
        
        
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.image('results/output5.png')   
        c1.write(' RMSE: 0.465')
        
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        c1.write(' ')
        
        c1.image('results/output6.png')  
        c1.write('RMSE: 1.508') 
    
    with st.container():
        c2.subheader('Zoomed Predictions')
        c2.image('results/output_zoom_1.png')
        c2.image('results/output_zoom_2.png')
        c2.image('results/output_zoom_3.png')
        c2.image('results/output_zoom_4.png')
        c2.image('results/output_zoom_5.png')
        c2.image('results/output_zoom_6.png')


st.link_button("Check out the Github Repo !", "https://github.com/adv-11/ml_mini_proj")
st.link_button("Check out the Report !", "https://github.com/adv-11/ml_mini_proj/blob/main/pdf/report.pdf")
pdf_file_path = "pdf/report.pdf"

with open(pdf_file_path, "rb") as pdf_file:

    pdf_bytes = pdf_file.read()
    
    st.download_button(label="Download Report !", data=pdf_bytes, file_name="report.pdf", mime="application/pdf")
    

