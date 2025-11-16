"""
===============================================================================
                    💳 BASELINE MODEL
                    
                 Первая модель БЕЗ обработки данных
                        "Quick and Dirty"
===============================================================================

ЦЕЛЬ:
-----
Создать базовую модель на СЫРЫХ данных чтобы потом сравнивать
насколько каждый шаг анализа улучшает результаты.

ЧТО ДЕЛАЕМ:
----------
1. Минимальная обработка (только чтобы модель запустилась)
2. Простая логистическая регрессия
3. Сохраняем метрики для сравнения
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, 
    classification_report,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# Пути 
import os
from pathlib import Path

# Ищем корень проекта по наличию папки data
current = Path(__file__).parent
while current != current.parent:
    if (current / 'data' / 'raw').exists():
        ROOT_DIR = current
        break
    current = current.parent
    
DATA_RAW = ROOT_DIR / 'data' / 'raw'
RESULTS = ROOT_DIR / 'results' / 'model_versions' / 'v0_baseline'
RESULTS.mkdir(parents=True, exist_ok=True) 

def create_baseline_model():
    """Создаем baseline модель на сырых данных."""
    
    print("="*60)
    print("BASELINE MODEL - v0")
    print("Модель на СЫРЫХ данных без обработки")
    print("="*60)
    
    # ========== 1. ЗАГРУЗКА ДАННЫХ ==========
    train_df = pd.read_csv(DATA_RAW / 'train.csv')
    print(f"\n📊 Загружено: {len(train_df)} записей")
    
    # ========== 2. МИНИМАЛЬНАЯ ПОДГОТОВКА ==========
    # Разделяем на X и y
    X = train_df.drop(['id', 'loan_status'], axis=1)
    y = train_df['loan_status']
    
    # Кодируем категориальные (иначе модель не запустится)
    print("\n🔧 Минимальная обработка категориальных...")
    
    le_dict = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le
        print(f"   • {col}: {len(le.classes_)} категорий")
    
    # НЕ обрабатываем выбросы (123 года)
    # НЕ делаем feature engineering
    # НЕ масштабируем
    print("\n⚠️ БЕЗ обработки:")
    print("   • Оставляем 123 в возрасте и стаже")
    print("   • Не масштабируем признаки")
    print("   • Не создаем новые признаки")
    
    # ========== 3. РАЗДЕЛЕНИЕ НА TRAIN/VAL ==========
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Разделение:")
    print(f"   Train: {len(X_train)} ({y_train.mean():.1%} positive)")
    print(f"   Val: {len(X_val)} ({y_val.mean():.1%} positive)")
    
    # ========== 4. ОБУЧЕНИЕ BASELINE ==========
    print("\n🚀 Обучаем Logistic Regression...")
    
    model = LogisticRegression(
        class_weight='balanced',  # Из-за дисбаланса
        max_iter=1000,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✅ Модель обучена!")
    
    # ========== 5. ПРЕДСКАЗАНИЯ ==========
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)
    
    # ========== 6. МЕТРИКИ ==========
    print("\n📈 МЕТРИКИ BASELINE:")
    print("-"*40)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification Report
    report = classification_report(y_val, y_pred, 
                                  target_names=['Rejected', 'Approved'])
    print("\nClassification Report:")
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
    
    # ========== 7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ==========
    
    # Сохраняем метрики в файл
    with open(RESULTS / 'metrics.txt', 'w') as f:
        f.write("BASELINE MODEL METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Model: LogisticRegression\n")
        f.write(f"Features: {X.shape[1]}\n")
        f.write(f"Train size: {len(X_train)}\n")
        f.write(f"Val size: {len(X_val)}\n")
        f.write("\nPERFORMANCE:\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write("\n" + report)
        f.write("\nCONFUSION MATRIX:\n")
        f.write(f"TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}\n")
        f.write(f"FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}\n")
        f.write("\nNOTES:\n")
        f.write("- No data cleaning (123 values kept)\n")
        f.write("- No feature engineering\n")
        f.write("- No scaling\n")
        f.write("- Only label encoding for categorical\n")
    
    # Визуализация Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Rejected', 'Approved'],
                yticklabels=['Rejected', 'Approved'])
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(RESULTS / 'confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Baseline Model')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(RESULTS / 'roc_curve.png')
    plt.close()
    
    # Feature Importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    importance.to_csv(RESULTS / 'feature_importance.csv', index=False)
    
    # Топ-10 важных признаков
    plt.figure(figsize=(10, 6))
    top10 = importance.head(10)
    plt.barh(range(10), top10['coefficient'].values)
    plt.yticks(range(10), top10['feature'].values)
    plt.xlabel('Coefficient')
    plt.title('Top 10 Features - Baseline Model')
    plt.grid(alpha=0.3)
    plt.savefig(RESULTS / 'top_features.png')
    plt.close()
    
    # Сохраняем модель
    with open(RESULTS / 'model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(RESULTS / 'label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)
    
    print(f"\n✅ Результаты сохранены в: {RESULTS}")
    
    # ========== 8. SUMMARY ==========
    print("\n" + "="*60)
    print("BASELINE SUMMARY:")
    print("="*60)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Precision (Approved): {cm[1,1]/(cm[1,1]+cm[0,1]):.3f}")
    print(f"Recall (Approved): {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")
    print("\n💡 Это наша отправная точка!")
    print("   Каждый следующий шаг должен улучшить эти метрики.")
    
    return model, roc_auc

if __name__ == "__main__":
    model, baseline_auc = create_baseline_model()
    
    # Сохраняем baseline метрику для сравнения
    with open(RESULTS.parent / 'baseline_auc.txt', 'w') as f:
        f.write(f"{baseline_auc:.4f}")
