# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

```
```c
data = pd.read_csv('/content/IOT-temp.xls')
print(data.head())
```
```c
data['noted_date'] = pd.to_datetime(data['noted_date'], format='%d-%m-%Y %H:%M', errors='coerce')
data['noted_date_numeric'] = (data['noted_date'] - data['noted_date'].min()).dt.total_seconds()
X = data[['noted_date_numeric']].values
y = data[['temp']].values
```
A - LINEAR TREND ESTIMATION
```c
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)
intercept_linear = linear_model.intercept_[0]
coefficient_linear = linear_model.coef_[0][0]
print(f"Linear Regression Formula: Temperature = {intercept_linear:.2f} + {coefficient_linear:.5f} * Time")

```
```c
plt.figure(figsize=(10, 6))
plt.scatter(data['noted_date'], y, label='Actual Temperature', color='blue')
plt.plot(data['noted_date'], y_pred_linear, color='red', label='Linear Trend')
plt.xlabel('noted_date')
plt.ylabel('temp')
plt.title('Linear Trend Estimation')
plt.legend()
plt.show()

```

B- POLYNOMIAL TREND ESTIMATION

```c
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)
intercept_poly = poly_model.intercept_[0]
coeffs_poly = poly_model.coef_[0]
print(f"Polynomial Regression Formula: Temperature = {intercept_poly:.2f} + "
      f"{coeffs_poly[1]:.5f} * Time + {coeffs_poly[2]:.10f} * Time^2")
```
```c
plt.figure(figsize=(10, 6))
plt.scatter(data['noted_date'], y, label='Actual Temperature', color='blue')
plt.plot(data['noted_date'], y_pred_poly, color='green', label='Polynomial Trend (degree=2)')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Polynomial Trend Estimation')
plt.legend()
plt.show()
```

### OUTPUT
A - LINEAR TREND ESTIMATION

![image](https://github.com/user-attachments/assets/5c075c4b-5d47-4aa9-8945-fab2e55c1040)
![download](https://github.com/user-attachments/assets/40c06c12-a55b-431f-bc05-40f027254eb6)


B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/673b32be-582b-4abc-a8b8-c0ebea6ca273)

![download](https://github.com/user-attachments/assets/a0db6c6b-83e3-4f23-b505-525c5d69e9ad)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
