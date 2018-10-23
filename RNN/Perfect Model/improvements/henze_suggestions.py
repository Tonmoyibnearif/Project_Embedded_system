# -*- codingb: utf-8 -*-
"""
Created on Tue Oct 23 12:39:26 2018

@author: x
"""

csv_file = "bla.csv"
data = read_in_data(csv)

normalized_data = normalize_data(data)
x_train,y_train,x_test,y_test = split_data(normalized_data,0.7,0.2,0.1)


rmse =[]

for shift in range(27):
    ar_x_train, ar_y_train =autoregression(x_train,y_train,shift)
    ar_x_test, ar_y_test =autoregression(x_test,y_test,shift)
    model = build_model()

    trained_model = train(model,x_train,y_train)
    rmse[shift] = evaluate(trained_model,x_test,y_test)
    rmse[shift] = 0.5

    predictions[shift] = predict(trained_model,x_val,y_val)
    predictions[shift] = [12371,233,45,6,7,8,2,35,9]