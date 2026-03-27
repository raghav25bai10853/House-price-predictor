#import the modules
import numpy as np
import pandas as pd

#house data
size_sqft = np.array([1400, 1600, 1700, 1875, 2200, 2350, 2600, 2700, 2970, 3200])
price_lakhs = np.array([25, 32, 29, 38, 19, 29, 45, 34, 40, 42])

#training the model (numpy linear regression)
X = size_sqft.reshape(-1, 1)
X_b = np.c_[np.ones((10, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(price_lakhs)
slope = theta_best[1]
intercept = theta_best[0]
print(f"trained model price = Rs{intercept:.0f}Lakh + Rs{slope:.3f}Lakhs/sqft")

#model dictionary
house_model = {
    'intercept': float(intercept),
    'slope': float(slope),
    'mean_size': float(np.mean(size_sqft)),
    'std_size': float(np.std(size_sqft))
}

#save to CSV (model params + predictions)
results_df = pd.DataFrame({
    'size_sqft': size_sqft,
    'actual_price_lakhs': price_lakhs,
    'predicted_price_lakhs': intercept + slope * size_sqft,
    'error_lakhs': np.abs(price_lakhs - (intercept + slope * size_sqft))
})

#add model params as extra rows
model_row = pd.DataFrame([house_model])
model_row['size_sqft'] = np.nan
model_row['actual_price_lakhs'] = np.nan  
model_row['predicted_price_lakhs'] = np.nan
model_row['error_lakhs'] = np.nan
final_df = pd.concat([results_df, model_row], ignore_index=True)
final_df.to_csv('house_price_model.csv', index=False)
print("saved: house_price_model.csv")
print("\n model CSV Preview:")
print(final_df.tail())

#load from CSV and predict
loaded_df = pd.read_csv('house_price_model.csv')
model_params = loaded_df[loaded_df['intercept'].notna()].iloc[0]
def predict_from_csv(size_sqft):
    return (model_params['intercept'] + 
            model_params['slope'] * size_sqft)

#predicting the prices for houses
new_houses = [1800, 2500, 3200]
for size in new_houses:
    pred = predict_from_csv(size)
    print(f"{size}sqft → Rs{pred:.0f}Lakhs")
