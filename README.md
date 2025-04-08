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

![image](https://github.com/user-attachments/assets/45114a9b-9b89-442d-9171-998885e8d66b)

![download](https://github.com/user-attachments/assets/e7204452-185c-4a20-ba9f-0d3b6c1c71ef)


B- POLYNOMIAL TREND ESTIMATION

![image](https://github.com/user-attachments/assets/9cb12760-914a-4216-a2a1-2a23c7f7d4d2)

![download](https://github.com/user-attachments/assets/14a31601-6cee-40a7-9923-1d3f0ca6e1e3)

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
