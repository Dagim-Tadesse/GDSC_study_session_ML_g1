import pandas as pd
from sklearn.linear_model import LinearRegression

pr = pd.read_csv("class_projects/csv/Housing.csv")

y = pr["price"]
x = pr[["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement",
       "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]]
x = pd.get_dummies(x, drop_first=True)

model = LinearRegression()
model.fit(x, y)

example_list = {
    "area": [1000],
    "bedrooms": [2],
    "bathrooms": [1],
    "stories": [1],
    "parking": [1],
    "mainroad_yes": [1],
    "guestroom_yes": [0],
    "basement_yes": [1],
    "hotwaterheating_yes": [0],
    "airconditioning_yes": [1],
    "prefarea_yes": [0],
    "furnishingstatus_semi-furnished": [1],
    "furnishingstatus_unfurnished": [0]
}

example = pd.DataFrame(example_list, columns=x.columns)

pred = model.predict(example)
print(f"Predicted price: ", int(pred[0]))
