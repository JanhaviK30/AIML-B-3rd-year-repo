{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOd8eFlJtBlMFr1k+LMBX2t",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JanhaviK30/AIML-B-3rd-year-repo/blob/main/personal%20project(models).py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Q_PLrtSMa5oz",
        "outputId": "6c7468f3-e0a0-448f-c2b4-2e30253d4cc3"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "File ‘heart_cleveland_upload.csv’ already there; not retrieving.\n",
            "\n",
            "File ‘heart.csv’ already there; not retrieving.\n",
            "\n",
            "Combined Dataset Shape (Rows, Columns): (1322, 15)\n",
            "Saved successfully as 'combined_heart_dataset.csv'!\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Download the necessary CSV files if they are not already present\n",
        "!wget -nc https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Project-on-Heart-Disease-Prediction/main/heart_cleveland_upload.csv\n",
        "!wget -nc https://raw.githubusercontent.com/reddyprasade/Machine-Learning-Project-on-Heart-Disease-Prediction/main/heart.csv\n",
        "\n",
        "df1 = pd.read_csv('heart_cleveland_upload.csv')\n",
        "df2 = pd.read_csv('heart.csv')\n",
        "\n",
        "\n",
        "combined_df = pd.concat([df1, df2], ignore_index=True)\n",
        "\n",
        "print(f\"Combined Dataset Shape (Rows, Columns): {combined_df.shape}\")\n",
        "\n",
        "combined_df.to_csv('combined_heart_dataset.csv', index=False)\n",
        "print(\"Saved successfully as 'combined_heart_dataset.csv'!\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df= pd.read_csv('combined_heart_dataset.csv')\n",
        "print(df.head)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "k3lbnrVccIRJ",
        "outputId": "66603273-5610-4da4-eabe-0580b45d5a13"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<bound method NDFrame.head of       age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  \\\n",
            "0      69    1   0       160   234    1        2      131      0      0.1   \n",
            "1      69    0   0       140   239    0        0      151      0      1.8   \n",
            "2      66    0   0       150   226    0        0      114      0      2.6   \n",
            "3      65    1   0       138   282    1        2      174      0      1.4   \n",
            "4      64    1   0       110   211    0        2      144      1      1.8   \n",
            "...   ...  ...  ..       ...   ...  ...      ...      ...    ...      ...   \n",
            "1317   59    1   1       140   221    0        1      164      1      0.0   \n",
            "1318   60    1   0       125   258    0        0      141      1      2.8   \n",
            "1319   47    1   0       110   275    0        0      118      1      1.0   \n",
            "1320   50    0   0       110   254    0        0      159      0      0.0   \n",
            "1321   54    1   0       120   188    0        1      113      0      1.4   \n",
            "\n",
            "      slope  ca  thal  condition  target  \n",
            "0         1   1     0        0.0     NaN  \n",
            "1         0   2     0        0.0     NaN  \n",
            "2         2   0     0        0.0     NaN  \n",
            "3         1   1     0        1.0     NaN  \n",
            "4         1   0     0        0.0     NaN  \n",
            "...     ...  ..   ...        ...     ...  \n",
            "1317      2   0     2        NaN     1.0  \n",
            "1318      1   1     3        NaN     0.0  \n",
            "1319      1   1     2        NaN     0.0  \n",
            "1320      2   0     2        NaN     1.0  \n",
            "1321      1   1     3        NaN     0.0  \n",
            "\n",
            "[1322 rows x 15 columns]>\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "missing_values = df.isnull().sum()\n",
        "print(missing_values)\n",
        "\n",
        "duplicate_count= df.duplicated().sum()\n",
        "print(duplicate_count)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "R0U07X0Vcdcl",
        "outputId": "b66aa1cd-4ec5-4e7d-f838-f1d74346ff1c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "age             0\n",
            "sex             0\n",
            "cp              0\n",
            "trestbps        0\n",
            "chol            0\n",
            "fbs             0\n",
            "restecg         0\n",
            "thalach         0\n",
            "exang           0\n",
            "oldpeak         0\n",
            "slope           0\n",
            "ca              0\n",
            "thal            0\n",
            "condition    1025\n",
            "target        297\n",
            "dtype: int64\n",
            "723\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "duplicate_count = df.duplicated().sum()\n",
        "if duplicate_count > 0:\n",
        "  df = df.drop_duplicates()\n",
        "  print(\"duplicates removed successfully\")\n",
        "print(df.info())\n",
        "print(df.describe())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Wn7hEBzRdTDz",
        "outputId": "b4d69fd1-70bc-452c-865c-c2fedcea11aa"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "duplicates removed successfully\n",
            "<class 'pandas.core.frame.DataFrame'>\n",
            "Index: 599 entries, 0 to 1175\n",
            "Data columns (total 15 columns):\n",
            " #   Column     Non-Null Count  Dtype  \n",
            "---  ------     --------------  -----  \n",
            " 0   age        599 non-null    int64  \n",
            " 1   sex        599 non-null    int64  \n",
            " 2   cp         599 non-null    int64  \n",
            " 3   trestbps   599 non-null    int64  \n",
            " 4   chol       599 non-null    int64  \n",
            " 5   fbs        599 non-null    int64  \n",
            " 6   restecg    599 non-null    int64  \n",
            " 7   thalach    599 non-null    int64  \n",
            " 8   exang      599 non-null    int64  \n",
            " 9   oldpeak    599 non-null    float64\n",
            " 10  slope      599 non-null    int64  \n",
            " 11  ca         599 non-null    int64  \n",
            " 12  thal       599 non-null    int64  \n",
            " 13  condition  297 non-null    float64\n",
            " 14  target     302 non-null    float64\n",
            "dtypes: float64(3), int64(12)\n",
            "memory usage: 74.9 KB\n",
            "None\n",
            "              age         sex          cp    trestbps        chol         fbs  \\\n",
            "count  599.000000  599.000000  599.000000  599.000000  599.000000  599.000000   \n",
            "mean    54.480801    0.679466    1.555927  131.647746  246.921536    0.146912   \n",
            "std      9.041481    0.467072    1.163748   17.647831   51.833011    0.354314   \n",
            "min     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000   \n",
            "25%     48.000000    0.000000    0.000000  120.000000  211.000000    0.000000   \n",
            "50%     56.000000    1.000000    2.000000  130.000000  242.000000    0.000000   \n",
            "75%     61.000000    1.000000    3.000000  140.000000  275.000000    0.000000   \n",
            "max     77.000000    1.000000    3.000000  200.000000  564.000000    1.000000   \n",
            "\n",
            "          restecg     thalach       exang     oldpeak       slope         ca  \\\n",
            "count  599.000000  599.000000  599.000000  599.000000  599.000000  599.00000   \n",
            "mean     0.759599  149.584307    0.327212    1.049249    1.003339    0.69783   \n",
            "std      0.827397   22.903224    0.469588    1.162814    0.733792    0.97314   \n",
            "min      0.000000   71.000000    0.000000    0.000000    0.000000    0.00000   \n",
            "25%      0.000000  133.000000    0.000000    0.000000    0.000000    0.00000   \n",
            "50%      1.000000  153.000000    0.000000    0.800000    1.000000    0.00000   \n",
            "75%      1.500000  166.000000    1.000000    1.600000    2.000000    1.00000   \n",
            "max      2.000000  202.000000    1.000000    6.200000    2.000000    4.00000   \n",
            "\n",
            "             thal   condition      target  \n",
            "count  599.000000  297.000000  302.000000  \n",
            "mean     1.580968    0.461279    0.543046  \n",
            "std      1.091027    0.499340    0.498970  \n",
            "min      0.000000    0.000000    0.000000  \n",
            "25%      0.000000    0.000000    0.000000  \n",
            "50%      2.000000    0.000000    1.000000  \n",
            "75%      2.000000    1.000000    1.000000  \n",
            "max      3.000000    1.000000    1.000000  \n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "\n",
        "# 1. Load the original raw files\n",
        "df1 = pd.read_csv('heart_cleveland_upload.csv')\n",
        "df2 = pd.read_csv('heart.csv')\n",
        "\n",
        "# 2. RENAME 'condition' to 'target' so the column structures line up exactly\n",
        "if 'condition' in df1.columns:\n",
        "    df1 = df1.rename(columns={'condition': 'target'})\n",
        "\n",
        "# 3. Combine them vertically now that they share the exact same 14 columns\n",
        "combined_df = pd.concat([df1, df2], ignore_index=True)\n",
        "\n",
        "# 4. Strip out duplicate entries\n",
        "before_dup = combined_df.shape[0]\n",
        "combined_df = combined_df.drop_duplicates()\n",
        "after_dup = combined_df.shape[0]\n",
        "\n",
        "print(\"==================================================\")\n",
        "print(\"          VERIFIED PREPROCESSING ENGINE          \")\n",
        "print(\"==================================================\")\n",
        "print(f\"✅ Removed {before_dup - after_dup} duplicate rows.\")\n",
        "print(f\"✅ Final Correct Matrix Shape: {combined_df.shape} (Must be 14 columns!)\")\n",
        "\n",
        "# 5. Sanity check for empty value gaps\n",
        "print(\"\\n🔍 Missing Values Summary:\")\n",
        "print(combined_df.isnull().sum())\n",
        "\n",
        "# 6. Save the true master clean data\n",
        "combined_df.to_csv('preprocessed_heart_dataset.csv', index=False)\n",
        "print(\"\\n🚀 Perfect dataset saved as 'preprocessed_heart_dataset.csv'!\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CYNPIOLlPj4q",
        "outputId": "c7840158-988a-459a-c319-7d2e9559cd2b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "==================================================\n",
            "          VERIFIED PREPROCESSING ENGINE          \n",
            "==================================================\n",
            "✅ Removed 723 duplicate rows.\n",
            "✅ Final Correct Matrix Shape: (599, 14) (Must be 14 columns!)\n",
            "\n",
            "🔍 Missing Values Summary:\n",
            "age         0\n",
            "sex         0\n",
            "cp          0\n",
            "trestbps    0\n",
            "chol        0\n",
            "fbs         0\n",
            "restecg     0\n",
            "thalach     0\n",
            "exang       0\n",
            "oldpeak     0\n",
            "slope       0\n",
            "ca          0\n",
            "thal        0\n",
            "target      0\n",
            "dtype: int64\n",
            "\n",
            "🚀 Perfect dataset saved as 'preprocessed_heart_dataset.csv'!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# 1. Load the freshly verified clean dataset\n",
        "df = pd.read_csv('preprocessed_heart_dataset.csv')\n",
        "X = df.drop(columns=['target'])\n",
        "y = df['target']\n",
        "\n",
        "# 2. Robust Stratified Train-Test Split using your requested Seed (31)\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=31, stratify=y\n",
        ")\n",
        "\n",
        "# 3. Feature Standardization (Mandatory for distance and boundary algorithms)\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "# 4. Define your exact requested model sequence with optimized configurations\n",
        "models = [\n",
        "    (\"Linear SVM\", SVC(kernel='linear', C=0.1, random_state=31)),\n",
        "    (\"RBF SVM\", SVC(kernel='rbf', C=1.0, gamma='scale', random_state=31)),\n",
        "    (\"Polynomial SVM\", SVC(kernel='poly', degree=3, C=1.0, random_state=31)),\n",
        "    (\"KNN\", KNeighborsClassifier(n_neighbors=30, weights='uniform', p=2)),\n",
        "    (\"Radius Neighbors\", RadiusNeighborsClassifier(radius=2.5, outlier_label='most_frequent'))\n",
        "]\n",
        "\n",
        "# 5. Execute Pipeline and Track Output\n",
        "print(\"==================================================\")\n",
        "print(\"     TRAINING RESULTS IN REQUESTED SEQUENCE       \")\n",
        "print(\"==================================================\\n\")\n",
        "\n",
        "for name, model in models:\n",
        "    # Train model\n",
        "    model.fit(X_train_scaled, y_train)\n",
        "\n",
        "    # Calculate performance metrics\n",
        "    train_acc = accuracy_score(y_train, model.predict(X_train_scaled)) * 100\n",
        "    test_acc = accuracy_score(y_test, model.predict(X_test_scaled)) * 100\n",
        "\n",
        "    print(f\"📊 {name}:\")\n",
        "    print(f\"   • Training Accuracy : {train_acc:.2f}%\")\n",
        "    print(f\"   • Testing Accuracy  : {test_acc:.2f}% (Main Accuracy)\")\n",
        "    print(f\"   • Overfitting Gap   : {abs(train_acc - test_acc):.2f}%\")\n",
        "    print(\"-\" * 40)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Erp6INEDP-MN",
        "outputId": "a6956e81-d99c-46e9-d171-eca866d8a4bf"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "==================================================\n",
            "     TRAINING RESULTS IN REQUESTED SEQUENCE       \n",
            "==================================================\n",
            "\n",
            "📊 Linear SVM:\n",
            "   • Training Accuracy : 75.99%\n",
            "   • Testing Accuracy  : 71.67% (Main Accuracy)\n",
            "   • Overfitting Gap   : 4.32%\n",
            "----------------------------------------\n",
            "📊 RBF SVM:\n",
            "   • Training Accuracy : 86.01%\n",
            "   • Testing Accuracy  : 80.00% (Main Accuracy)\n",
            "   • Overfitting Gap   : 6.01%\n",
            "----------------------------------------\n",
            "📊 Polynomial SVM:\n",
            "   • Training Accuracy : 86.43%\n",
            "   • Testing Accuracy  : 77.50% (Main Accuracy)\n",
            "   • Overfitting Gap   : 8.93%\n",
            "----------------------------------------\n",
            "📊 KNN:\n",
            "   • Training Accuracy : 79.75%\n",
            "   • Testing Accuracy  : 80.00% (Main Accuracy)\n",
            "   • Overfitting Gap   : 0.25%\n",
            "----------------------------------------\n",
            "📊 Radius Neighbors:\n",
            "   • Training Accuracy : 87.68%\n",
            "   • Testing Accuracy  : 75.00% (Main Accuracy)\n",
            "   • Overfitting Gap   : 12.68%\n",
            "----------------------------------------\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.ensemble import VotingClassifier\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "# 1. Load Data and split\n",
        "df = pd.read_csv('preprocessed_heart_dataset.csv')\n",
        "X = df.drop(columns=['target'])\n",
        "y = df['target']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.2, random_state=31, stratify=y\n",
        ")\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = scaler.fit_transform(X_train)\n",
        "X_test_scaled = scaler.transform(X_test)\n",
        "\n",
        "print(\"=====================================================================\")\n",
        "print(\"         FAST OPTIMIZED ENSEMBLE ENGINE (FIXED SEED 31)             \")\n",
        "print(\"=====================================================================\\n\")\n",
        "\n",
        "# 2. Define Core Models\n",
        "rbf_svm = SVC(kernel='rbf', probability=True, random_state=31)\n",
        "poly_svm = SVC(kernel='poly', degree=3, probability=True, random_state=31)\n",
        "knn = KNeighborsClassifier()\n",
        "\n",
        "# Combine them into a Soft-Voting Ensemble\n",
        "ensemble = VotingClassifier(\n",
        "    estimators=[\n",
        "        ('rbf', rbf_svm),\n",
        "        ('poly', poly_svm),\n",
        "        ('knn', knn)\n",
        "    ],\n",
        "    voting='soft'\n",
        ")\n",
        "\n",
        "# 3. Fast Targeted Grid Search (Perfected with solid numbers)\n",
        "param_grid = {\n",
        "    'rbf__C': [0.1, 1.0, 5.0],\n",
        "    'rbf__gamma': ['scale', 'auto'],\n",
        "    'poly__C': [0.1, 1.0],\n",
        "    'knn__n_neighbors': [5, 10, 15, 20, 25, 30],\n",
        "    'knn__weights': ['uniform']\n",
        "}\n",
        "\n",
        "# 4. Execute Optimization\n",
        "grid_search = GridSearchCV(\n",
        "    estimator=ensemble,\n",
        "    param_grid=param_grid,\n",
        "    cv=3,\n",
        "    scoring='accuracy',\n",
        "    n_jobs=-1\n",
        ")\n",
        "\n",
        "grid_search.fit(X_train_scaled, y_train)\n",
        "\n",
        "# 5. Performance Metrics Reporting\n",
        "optimized_ensemble = grid_search.best_estimator_\n",
        "final_train_acc = accuracy_score(y_train, optimized_ensemble.predict(X_train_scaled)) * 100\n",
        "final_test_acc = accuracy_score(y_test, optimized_ensemble.predict(X_test_scaled)) * 100\n",
        "\n",
        "print(\"=====================================================================\")\n",
        "print(\"                    OPTIMIZED ENSEMBLE RESOLUTION                    \")\n",
        "print(\"=====================================================================\")\n",
        "print(f\"📊 Ensemble Training Accuracy : {final_train_acc:.2f}%\")\n",
        "print(f\"🚀 ENSEMBLE TESTING ACCURACY  : {final_test_acc:.2f}% (Main Accuracy)\")\n",
        "print(f\"📉 Optimized Overfitting Gap  : {abs(final_train_acc - final_test_acc):.2f}%\")\n",
        "print(\"=====================================================================\")\n",
        "\n",
        "print(\"\\n📋 Detailed Clinical Classification Report:\")\n",
        "print(classification_report(y_test, optimized_ensemble.predict(X_test_scaled)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "FJUJZSnwfRXI",
        "outputId": "59e3bc0e-3e57-423e-dd04-055f4f79159f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "=====================================================================\n",
            "         FAST OPTIMIZED ENSEMBLE ENGINE (FIXED SEED 31)             \n",
            "=====================================================================\n",
            "\n",
            "=====================================================================\n",
            "                    OPTIMIZED ENSEMBLE RESOLUTION                    \n",
            "=====================================================================\n",
            "📊 Ensemble Training Accuracy : 82.25%\n",
            "🚀 ENSEMBLE TESTING ACCURACY  : 80.00% (Main Accuracy)\n",
            "📉 Optimized Overfitting Gap  : 2.25%\n",
            "=====================================================================\n",
            "\n",
            "📋 Detailed Clinical Classification Report:\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.78      0.83      0.81        60\n",
            "           1       0.82      0.77      0.79        60\n",
            "\n",
            "    accuracy                           0.80       120\n",
            "   macro avg       0.80      0.80      0.80       120\n",
            "weighted avg       0.80      0.80      0.80       120\n",
            "\n"
          ]
        }
      ]
    }
  ]
}