# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from ucimlrepo import fetch_ucirepo

# Cargar el dataset Iris
iris = fetch_ucirepo(id=53)

# Obtener los datos
X = iris.data.features
y = iris.data.targets

# Verifica la forma de 'y' (debería ser un DataFrame o un array)
print(f"Forma de 'y' antes de aplanar: {y.shape}")

# Si 'y' es un DataFrame, convertirlo a un array de numpy y luego a un array unidimensional
if isinstance(y, pd.DataFrame):
    y = y.values.flatten()  # Convierte 'y' a un array numpy unidimensional

# Convertir los datos en un DataFrame para inspección
df = pd.DataFrame(X, columns=iris.feature_names)

# 1. Revisión de valores nulos
print("Valores nulos por columna:")
print(df.isnull().sum())

# 2. Revisar duplicados
print("\nNúmero de duplicados:", df.duplicated().sum())

# 3. Revisar distribución de las clases
print("\nDistribución de las clases (target):")
print(pd.Series(y).value_counts())

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Asegurarse de que 'y_train' y 'y_test' estén unidimensionales
y_train = y_train.flatten()  # Convierte 'y_train' a un array unidimensional
y_test = y_test.flatten()    # Convierte 'y_test' a un array unidimensional

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Iniciar el seguimiento con MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.start_run()

# Registrar parámetros del modelo
mlflow.log_param("n_estimators", 100)

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Hacer predicciones y evaluar
y_pred = model.predict(X_test_scaled)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy}")

# Imprimir reporte de clasificación (precisión, recall, F1-score)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Imprimir matriz de confusión
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

# Registrar métricas del modelo
mlflow.log_metric("accuracy", accuracy)

# Registrar reporte de clasificación como texto
mlflow.log_text(str(classification_report(y_test, y_pred)), "classification_report.txt")

# Registrar el modelo con ejemplo de entrada
input_example = X_test_scaled[0].reshape(1, -1)
mlflow.sklearn.log_model(model, "model", input_example=input_example)

# Finalizar el seguimiento
mlflow.end_run()
