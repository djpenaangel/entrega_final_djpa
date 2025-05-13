# tests/test_train.py

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def test_model_training():
    # Simulamos datos de entrada
    X = np.random.rand(150, 4)
    y = np.random.choice([0, 1, 2], size=150)
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Verificar si el modelo hace predicciones
    assert model.predict(X_test_scaled).shape == (30,)
